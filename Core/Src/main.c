/* main.c — Pyboard v1.1 (STM32F405) — AWG + ACCEL + SPI/FPGA + CLI
 *
 * Recursos ativos nesta versão:
 *  - DAC CH1 (PA4) dirigido por TIM2->TRGO (UPDATE) via DMA circular + LUT (N=256).
 *  - Janelas (“taper”): NONE / HANN / BLACKMAN / NUTTALL.
 *  - MMA7660 (I2C1) com filtro de Kalman (ativável).
 *  - Heartbeat (LED azul) ON/OFF via SYS HB <0|1>.
 *  - SPI1 mestre → FPGA (upload com CRC32 periférico).
 *  - CLI robusta (tabela de comandos).
 *
 * Observação: ADC1/2/3 foram DESABILITADOS nesta build para evitar conflitos.
 */

#include "main.h"
#include "usb_device.h"
#include "usbd_cdc_if.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#include "accel_mma7660.h"   // driver do MMA7660

/* ============================ Handles periféricos ============================ */
CRC_HandleTypeDef hcrc;

DAC_HandleTypeDef  hdac;
/* O DMA do DAC é configurado no MSP e linkado via __HAL_LINKDMA(&hdac, DMA_Handle1, hdma_dac1); */

I2C_HandleTypeDef  hi2c1;
SPI_HandleTypeDef  hspi1;

DAC_HandleTypeDef  hdac;
DMA_HandleTypeDef  hdma_dac1;   // <= ADICIONE ESTA LINHA


TIM_HandleTypeDef  htim2;  // DAC (TRGO)
TIM_HandleTypeDef  htim3;  // (reservado; não usado sem ADC)

/* usbd_cdc_if.c */
extern USBD_HandleTypeDef hUsbDeviceFS;

/* CLI (buffers preenchidos em usbd_cdc_if.c) */
extern volatile uint8_t  cdc_cmd_buffer[128];
extern volatile uint32_t cdc_cmd_length;
extern volatile uint8_t  cdc_cmd_ready;

/* ============================ Estado global ============================ */
volatile AppState g_app_state = STATE_IDLE;   // definido em main.h (ex.: enum AppState)

static volatile uint8_t g_hb_enable = 1;      // heartbeat (LED azul)

/* ============================ Kalman (3 eixos) ============================ */
typedef struct { float Q, R, x, P; } KFilter;
static KFilter kx = {.Q=0.02f,.R=0.8f,.x=0,.P=1};
static KFilter ky = {.Q=0.02f,.R=0.8f,.x=0,.P=1};
static KFilter kz = {.Q=0.02f,.R=0.8f,.x=0,.P=1};
static uint8_t g_kalman_on = 1;

static inline float kalman_step(KFilter *kf, float z){
  kf->P += kf->Q;
  float K = kf->P / (kf->P + kf->R);
  kf->x = kf->x + K*(z - kf->x);
  kf->P = (1.f - K)*kf->P;
  return kf->x;
}

/* ============================ AWG / LUT / Janela ============================ */
#define LUT_N               256
static uint16_t s_lut[LUT_N];
static const float DAC_FS_MAX_HZ = 1.0e6f;

typedef enum { WT_SINE, WT_SQUARE, WT_TRI, WT_SAWUP, WT_SAWDN } WaveType;
typedef enum { WIN_NONE, WIN_HANN, WIN_BLACKMAN, WIN_NUTTALL } WinType;
static WaveType g_wave = WT_SINE;
static WinType  g_win  = WIN_NONE;
static float    g_taper_percent = 50.f; // 0..100

static inline float fmax_from_fs(float fs, int N){ return fs / (float)N; }

static void apply_window(float *w, int N, WinType wt, float taper_pct){
  if (wt==WIN_NONE || taper_pct<=0.1f){ for(int i=0;i<N;i++) w[i]=1.0f; return; }
  float M=(float)(N-1);
  for(int n=0;n<N;n++){
    float a = (float)n/M;
    float mult=1.f;
    switch(wt){
      case WIN_HANN:     mult = 0.5f*(1.f - cosf(2.f*M_PI*a)); break;
      case WIN_BLACKMAN: mult = 0.42f - 0.5f*cosf(2.f*M_PI*a) + 0.08f*cosf(4.f*M_PI*a); break;
      case WIN_NUTTALL:  mult = 0.355768f - 0.487396f*cosf(2.f*M_PI*a) + 0.144232f*cosf(4.f*M_PI*a) - 0.012604f*cosf(6.f*M_PI*a); break;
      default: break;
    }
    float edge = taper_pct/100.f, t=1.f;
    if (a<edge) t=a/edge; else if (a>(1.f-edge)) t=(1.f-a)/edge;
    w[n] = 1.f - (1.f - mult)*t;
  }
}

static void fill_lut(WaveType wt){
  float win[LUT_N]; apply_window(win, LUT_N, g_win, g_taper_percent);
  for (int i=0;i<LUT_N;i++){
    float ph=(float)i/(float)LUT_N, y=0.f;
    switch(wt){
      case WT_SINE:   y = 0.5f*(sinf(2.f*M_PI*ph)+1.f); break;
      case WT_SQUARE: y = (ph<0.5f)?1.f:0.f; break;
      case WT_TRI:    y = (ph<0.5f)?(ph*2.f):(2.f-2.f*ph); break;
      case WT_SAWUP:  y = ph; break;
      case WT_SAWDN:  y = 1.f - ph; break;
    }
    y *= win[i];
    float v = y*4095.f; if (v<0) v=0; if (v>4095.f) v=4095.f;
    s_lut[i] = (uint16_t)(v+0.5f);
  }
}

static bool dac_start(float freq){
  if (freq<=0) return false;
  float fs = (float)LUT_N * freq; if (fs>DAC_FS_MAX_HZ) fs=DAC_FS_MAX_HZ;

  uint32_t pclk1  = HAL_RCC_GetPCLK1Freq();     // ~42 MHz
  uint32_t timclk = pclk1*2U; if (!timclk) timclk=84000000U;

  uint32_t psc=0;
  uint32_t arr = (uint32_t)((float)timclk/fs); if (arr<1) arr=1; arr-=1;

  __HAL_TIM_DISABLE(&htim2);
  __HAL_TIM_SET_PRESCALER(&htim2, psc);
  __HAL_TIM_SET_AUTORELOAD(&htim2, arr);
  __HAL_TIM_SET_COUNTER(&htim2, 0);

  /* (Re)gera a LUT conforme tipo de onda atual */
  fill_lut(g_wave);

  /* IMPORTANTE: requer hdma_dac1 linkado no MSP (DMA1 Stream5, Channel 7) */
  if (HAL_DAC_Start_DMA(&hdac, DAC_CHANNEL_1, (uint32_t*)s_lut, LUT_N, DAC_ALIGN_12B_R)!=HAL_OK)
    return false;

  if (HAL_TIM_Base_Start(&htim2)!=HAL_OK){
    HAL_DAC_Stop_DMA(&hdac, DAC_CHANNEL_1);
    return false;
  }
  return true;
}

static void dac_stop(void){
  HAL_TIM_Base_Stop(&htim2);
  HAL_DAC_Stop_DMA(&hdac, DAC_CHANNEL_1);
}

/* ============================ Heartbeat ============================ */
static inline void put_prompt(void){ printf("> "); }
static void hb_tick(void){
  static uint32_t t0=0; if (!g_hb_enable) return;
  if (HAL_GetTick()-t0>500){ t0=HAL_GetTick(); HAL_GPIO_TogglePin(LED_BLUE_GPIO_Port, LED_BLUE_Pin); }
}

/* ============================ SPI / FPGA Upload ============================ */
#define FPGA_MAX_BYTES     (32u*1024u*1024u)
#define RX_BIN_BUF_SZ      2048u

static volatile uint8_t  g_bin_mode       = 0;
static volatile uint32_t g_bin_bytes_total= 0;
static volatile uint32_t g_bin_bytes_left = 0;
static volatile uint32_t g_bin_crc_expect = 0;
static volatile uint32_t g_bin_crc_calc   = 0;

static uint8_t  g_rx_bin_buf[RX_BIN_BUF_SZ];
static volatile uint32_t g_rx_bin_w=0, g_rx_bin_r=0;

static inline void fpga_cs_low(void){ HAL_GPIO_WritePin(FPGA_CS_GPIO_Port, FPGA_CS_Pin, GPIO_PIN_RESET); }
static inline void fpga_cs_high(void){ HAL_GPIO_WritePin(FPGA_CS_GPIO_Port, FPGA_CS_Pin, GPIO_PIN_SET); }

static void spi_set_prescaler(uint32_t br_prescaler){
  hspi1.Init.BaudRatePrescaler = br_prescaler;
  if (HAL_SPI_Init(&hspi1)!=HAL_OK) Error_Handler();
}

static void bin_reset_ring(void){ __disable_irq(); g_rx_bin_w=g_rx_bin_r=0; __enable_irq(); }
static uint32_t bin_ring_avail(void){ uint32_t w=g_rx_bin_w,r=g_rx_bin_r; return (w>=r)?(w-r):(RX_BIN_BUF_SZ-r+w); }

static uint32_t bin_ring_read(uint8_t *dst, uint32_t want){
  uint32_t got=0; __disable_irq();
  while (got<want && g_rx_bin_r!=g_rx_bin_w){
    dst[got++]=g_rx_bin_buf[g_rx_bin_r++]; if (g_rx_bin_r>=RX_BIN_BUF_SZ) g_rx_bin_r=0;
  }
  __enable_irq(); return got;
}

/* Chamado pelo usbd_cdc_if.c quando App_IsBinaryMode()=1 */
void CDC_OnRxData(uint8_t *buf, uint32_t len){
  if (!g_bin_mode || len==0) return;
  for (uint32_t i=0;i<len;i++){
    uint32_t next=g_rx_bin_w+1; if (next>=RX_BIN_BUF_SZ) next=0;
    if (next==g_rx_bin_r) break; // overflow (descarta)
    g_rx_bin_buf[g_rx_bin_w]=buf[i]; g_rx_bin_w=next;
  }
}
uint8_t App_IsBinaryMode(void){ return g_bin_mode; }

static void process_fpga_upload(void){
  if (!g_bin_mode) return;
  static uint8_t chunk[1024];

  uint32_t need = (g_bin_bytes_left>sizeof(chunk))?sizeof(chunk):g_bin_bytes_left;
  if (bin_ring_avail()<need) return; // aguarda mais bytes

  uint32_t got = bin_ring_read(chunk, need); if (!got) return;

  // CRC periférico (palavras de 32b; faz padding do resto)
  uint32_t w=got/4, rem=got%4;
  if (w) HAL_CRC_Accumulate(&hcrc,(uint32_t*)chunk,w);
  if (rem){ uint8_t pad[4]={0,0,0,0}; for (uint32_t i=0;i<rem;i++) pad[i]=chunk[4*w+i]; HAL_CRC_Accumulate(&hcrc,(uint32_t*)pad,1); }

  if (HAL_SPI_Transmit(&hspi1, chunk, got, HAL_MAX_DELAY)!=HAL_OK){
    printf("ERROR: SPI TX\r\n"); g_bin_mode=0; fpga_cs_high(); put_prompt(); return;
  }

  g_bin_bytes_left -= got;
  if (g_bin_bytes_left==0){
    g_bin_crc_calc = hcrc.Instance->DR;
    fpga_cs_high(); g_bin_mode=0;
    if (g_bin_crc_calc==g_bin_crc_expect) printf("FPGA_UPLOAD_OK\r\n");
    else printf("FPGA_UPLOAD_BADCRC exp:%lu got:%lu\r\n",(unsigned long)g_bin_crc_expect,(unsigned long)g_bin_crc_calc);
    put_prompt();
  }
}

/* ============================ CLI ============================ */
typedef void (*CmdFn)(int argc, char **argv);
typedef struct { const char *name; CmdFn fn; const char *help; } Cmd;

static void cli_help(int argc, char**argv);
static void cli_ping(int argc, char**argv);
static void cli_led(int argc, char**argv);
static void cli_accel(int argc, char**argv);
static void cli_kalman(int argc, char**argv);
static void cli_kalman_set(int argc, char**argv);
static void cli_dac(int argc, char**argv);
static void cli_wave(int argc, char**argv);
static void cli_wavewin(int argc, char**argv);
static void cli_sys(int argc, char**argv);
/* ADCs desabilitados */
static void cli_adc_start(int argc, char**argv);
static void cli_adc_stop(int argc, char**argv);
static void cli_adc_cfg(int argc, char**argv);
static void cli_adc_read(int argc, char**argv);

static void cli_spi_speed(int argc, char**argv);
static void cli_fpga_cs(int argc, char**argv);
static void cli_fpga_upload(int argc, char**argv);
static void cli_fpga_abort(int argc, char**argv);
static void cli_fpga_peek(int argc, char**argv);
static void cli_noop(int argc, char**argv);

static const Cmd cmds[] = {
  {"HELP",      cli_help,      "Lista de comandos"},
  {"PING",      cli_ping,      "PONG"},

  {"LED",       cli_led,       "LED <n:0..3> <0|1>"},
  {"ACCEL",     cli_accel,     "ACCEL <0|1> (stream A:x,y,z)"},
  {"KALMAN",    cli_kalman,    "KALMAN <0|1> habilita filtro"},
  {"KALMAN_SET",cli_kalman_set,"KALMAN_SET <Q> <R>"},

  {"DAC",       cli_dac,       "DAC <freq|0> (seno compat.)"},
  {"WAVE",      cli_wave,      "WAVE <SINE|SQUARE|TRI|SAWUP|SAWDN> <freq>"},
  {"WAVEWIN",   cli_wavewin,   "WAVEWIN <NONE|HANN|BLACKMAN|NUTTALL> <taper%>"},

  {"SYS",       cli_sys,       "SYS INFO | SYS HB <0|1> | SYS RESET"},

  /* ADCs desabilitados nesta build */
  {"ADC_START", cli_adc_start, "ADC_START (DISABLED)"},
  {"ADC_STOP",  cli_adc_stop,  "ADC_STOP (DISABLED)"},
  {"ADC_CFG",   cli_adc_cfg,   "ADC_CFG ... (DISABLED)"},
  {"ADC_READ",  cli_adc_read,  "ADC_READ ... (DISABLED)"},

  {"SPI_SPEED", cli_spi_speed, "SPI_SPEED <2|4|8|16|32|64|128|256>"},
  {"FPGA_CS",   cli_fpga_cs,   "FPGA_CS <0|1>"},
  {"FPGA_UPLOAD",cli_fpga_upload,"FPGA_UPLOAD <bytes> <crc32>"},
  {"FPGA_ABORT",cli_fpga_abort,"Cancela upload binario"},
  {"FPGA_PEEK", cli_fpga_peek, "FPGA_PEEK <nbytes>"},
  {"NOOP",      cli_noop,      "No operation"},
};

static void cli_help(int argc, char**argv){
  printf("Comandos:\r\n");
  for (unsigned i=0;i<sizeof(cmds)/sizeof(cmds[0]);i++)
    printf("  %-12s %s\r\n", cmds[i].name, cmds[i].help);
}
static void cli_ping(int argc, char**argv){ printf("PONG\r\n"); }

static void cli_led(int argc, char**argv){
  if (argc<3){ printf("ERROR: LED <n> <0|1>\r\n"); return; }
  int n=atoi(argv[1]), v=atoi(argv[2]);
  GPIO_TypeDef *port=NULL; uint16_t pin=0;
  switch(n){
    case 0: port=LED_RED_GPIO_Port;    pin=LED_RED_Pin;    break;
    case 1: port=LED_GREEN_GPIO_Port;  pin=LED_GREEN_Pin;  break;
    case 2: port=LED_YELLOW_GPIO_Port; pin=LED_YELLOW_Pin; break;
    case 3: port=LED_BLUE_GPIO_Port;   pin=LED_BLUE_Pin;   break;
    default: printf("ERROR: LED invalido\r\n"); return;
  }
  HAL_GPIO_WritePin(port,pin,v?GPIO_PIN_SET:GPIO_PIN_RESET);
  printf("OK\r\n");
}

static void cli_accel(int argc, char**argv){
  if (argc<2){ printf("ERROR: ACCEL <0|1>\r\n"); return; }
  int on=atoi(argv[1]);
  if (on){
    if (!MMA7660_Init(&hi2c1, MMA_AVDD_GPIO_Port, MMA_AVDD_Pin)){ printf("ERROR: MMA7660 init\r\n"); return; }
    g_app_state=STATE_STREAMING_ACCEL;
  } else g_app_state=STATE_IDLE;
  printf("OK\r\n");
}

static void cli_kalman(int argc, char**argv){
  if (argc<2){ printf("ERROR: KALMAN <0|1>\r\n"); return; }
  g_kalman_on=(uint8_t)atoi(argv[1]); printf("OK\r\n");
}
static void cli_kalman_set(int argc, char**argv){
  if (argc<3){ printf("ERROR: KALMAN_SET <Q> <R>\r\n"); return; }
  float Q=strtof(argv[1],NULL), R=strtof(argv[2],NULL);
  kx.Q=ky.Q=kz.Q=Q; kx.R=ky.R=kz.R=R; printf("OK\r\n");
}

static void cli_dac(int argc, char**argv){
  if (argc<2){ printf("ERROR: DAC <freq|0>\r\n"); return; }
  float f=strtof(argv[1],NULL);
  if (f>0.f){
    g_wave=WT_SINE;
    float fmax=fmax_from_fs(DAC_FS_MAX_HZ, LUT_N); if (f>fmax) f=fmax;
    if (dac_start(f)) printf("OK\r\n"); else printf("ERROR: DAC start\r\n");
  } else { dac_stop(); printf("OK\r\n"); }
}

static void cli_wave(int argc, char**argv){
  if (argc<3){ printf("ERROR: WAVE <SINE|SQUARE|TRI|SAWUP|SAWDN> <freq>\r\n"); return; }
  float f=strtof(argv[2],NULL); if (f<=0){ printf("ERROR: freq invalida\r\n"); return; }
  if      (!strcmp(argv[1],"SINE"))   g_wave=WT_SINE;
  else if (!strcmp(argv[1],"SQUARE")) g_wave=WT_SQUARE;
  else if (!strcmp(argv[1],"TRI"))    g_wave=WT_TRI;
  else if (!strcmp(argv[1],"SAWUP"))  g_wave=WT_SAWUP;
  else if (!strcmp(argv[1],"SAWDN"))  g_wave=WT_SAWDN;
  else { printf("ERROR: tipo invalido\r\n"); return; }
  float fmax=fmax_from_fs(DAC_FS_MAX_HZ, LUT_N); if (f>fmax) f=fmax;
  if (dac_start(f)) printf("OK\r\n"); else printf("ERROR: DAC start\r\n");
}

static void cli_wavewin(int argc, char**argv){
  if (argc<3){ printf("ERROR: WAVEWIN <NONE|HANN|BLACKMAN|NUTTALL> <taper%%>\r\n"); return; }
  if      (!strcmp(argv[1],"NONE"))     g_win=WIN_NONE;
  else if (!strcmp(argv[1],"HANN"))     g_win=WIN_HANN;
  else if (!strcmp(argv[1],"BLACKMAN")) g_win=WIN_BLACKMAN;
  else if (!strcmp(argv[1],"NUTTALL"))  g_win=WIN_NUTTALL;
  else { printf("ERROR: janela invalida\r\n"); return; }
  float t=strtof(argv[2],NULL); if(t<0) t=0; if(t>100) t=100; g_taper_percent=t;
  printf("OK\r\n");
}

static void cli_sys(int argc, char**argv){
  if (argc>=2){
    if (!strcmp(argv[1],"RESET")){ printf("RESETTING...\r\n"); HAL_Delay(20); NVIC_SystemReset(); return; }
    if (!strcmp(argv[1],"HB") && argc>=3){
      g_hb_enable=(uint8_t)atoi(argv[2]); if (!g_hb_enable) HAL_GPIO_WritePin(LED_BLUE_GPIO_Port, LED_BLUE_Pin, GPIO_PIN_RESET);
      printf("OK\r\n"); return;
    }
  }
  printf("SYS: SYSCLK=%lu, PCLK1=%lu, PCLK2=%lu, HB=%u, fmax=%.1fHz\r\n",
    HAL_RCC_GetSysClockFreq(), HAL_RCC_GetPCLK1Freq(), HAL_RCC_GetPCLK2Freq(),
    (unsigned)g_hb_enable, fmax_from_fs(DAC_FS_MAX_HZ,LUT_N));
  printf("OK\r\n");
}

/* ADCs desabilitados */
static void cli_adc_start(int argc, char**argv){ printf("DISABLED\r\n"); }
static void cli_adc_stop (int argc, char**argv){ printf("DISABLED\r\n"); }
static void cli_adc_cfg  (int argc, char**argv){ printf("DISABLED\r\n"); }
static void cli_adc_read (int argc, char**argv){ printf("DISABLED\r\n"); }

static void cli_spi_speed(int argc, char**argv){
  if (argc<2){ printf("ERROR: SPI_SPEED <2|4|8|16|32|64|128|256>\r\n"); return; }
  int d=atoi(argv[1]); uint32_t p=SPI_BAUDRATEPRESCALER_8;
  switch(d){
    case 2: p=SPI_BAUDRATEPRESCALER_2; break; case 4: p=SPI_BAUDRATEPRESCALER_4; break;
    case 8: p=SPI_BAUDRATEPRESCALER_8; break; case 16:p=SPI_BAUDRATEPRESCALER_16; break;
    case 32:p=SPI_BAUDRATEPRESCALER_32; break; case 64:p=SPI_BAUDRATEPRESCALER_64; break;
    case 128:p=SPI_BAUDRATEPRESCALER_128; break; case 256:p=SPI_BAUDRATEPRESCALER_256; break;
    default: printf("ERROR: divisor invalido\r\n"); return;
  }
  spi_set_prescaler(p); printf("OK\r\n");
}
static void cli_fpga_cs(int argc, char**argv){
  if (argc<2){ printf("ERROR: FPGA_CS <0|1>\r\n"); return; }
  int v=atoi(argv[1]); if (v) fpga_cs_low(); else fpga_cs_high(); printf("OK\r\n");
}
static void cli_fpga_upload(int argc, char**argv){
  if (argc<3){ printf("ERROR: FPGA_UPLOAD <bytes> <crc32>\r\n"); return; }
  uint32_t n=(uint32_t)strtoul(argv[1],NULL,0), c=(uint32_t)strtoul(argv[2],NULL,0);
  if (!n || n>FPGA_MAX_BYTES){ printf("ERROR: tamanho invalido\r\n"); return; }
  __HAL_CRC_DR_RESET(&hcrc);
  fpga_cs_low(); bin_reset_ring();
  g_bin_bytes_total=n; g_bin_bytes_left=n; g_bin_crc_expect=c; g_bin_mode=1;
  printf("FPGA_UPLOAD_READY\r\n");
}
static void cli_fpga_abort(int argc, char**argv){
  if (g_bin_mode){ g_bin_mode=0; fpga_cs_high(); } printf("OK\r\n");
}
static void cli_fpga_peek(int argc, char**argv){
  if (argc<2){ printf("ERROR: FPGA_PEEK <nbytes>\r\n"); return; }
  uint32_t n=(uint32_t)strtoul(argv[1],NULL,0); if (!n||n>1024){ printf("ERROR: limite 1024\r\n"); return; }
  uint8_t tmp[1024]; memset(tmp,0xFF,n);
  if (HAL_SPI_TransmitReceive(&hspi1,tmp,tmp,n,HAL_MAX_DELAY)!=HAL_OK){ printf("ERROR: SPI TR\r\n"); return; }
  for (uint32_t i=0; i<n; i++) { printf("%02X", tmp[i]); }
  printf("\r\nOK\r\n");
}

static void cli_noop(int argc, char**argv){ printf("OK\r\n"); }

/* Parser e dispatch */
static void CLI_ProcessLine(const char* line){
  char buf[128]; strncpy(buf,line,sizeof(buf)-1); buf[sizeof(buf)-1]='\0';
  char *argv[12]; int argc=0; char *tok=strtok(buf," \t\r\n");
  while (tok && argc<12){ argv[argc++]=tok; tok=strtok(NULL," \t\r\n"); }
  if (argc==0) return;
  for (unsigned i=0;i<sizeof(cmds)/sizeof(cmds[0]);i++){
    if (strcmp(argv[0],cmds[i].name)==0){ cmds[i].fn(argc,argv); return; }
  }
  printf("ERROR: comando desconhecido. Use HELP\r\n");
}

/* ============================ System/Init ============================ */
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_DAC_Init(void);
static void MX_I2C1_Init(void);
static void MX_SPI1_Init(void);
static void MX_CRC_Init(void);
static void MX_TIM2_Init(void);
static void MX_TIM3_Init(void);   // não usado sem ADC
// ADCs propositalmente omitidos

void SystemClock_Config(void){
  RCC_OscInitTypeDef RCC_Osc={0};
  RCC_ClkInitTypeDef RCC_Clk={0};
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  RCC_Osc.OscillatorType=RCC_OSCILLATORTYPE_HSE;
  RCC_Osc.HSEState=RCC_HSE_ON;
  RCC_Osc.PLL.PLLState=RCC_PLL_ON;
  RCC_Osc.PLL.PLLSource=RCC_PLLSOURCE_HSE;
  RCC_Osc.PLL.PLLM=12; RCC_Osc.PLL.PLLN=336; RCC_Osc.PLL.PLLP=RCC_PLLP_DIV2; RCC_Osc.PLL.PLLQ=7;
  if (HAL_RCC_OscConfig(&RCC_Osc)!=HAL_OK) Error_Handler();

  RCC_Clk.ClockType=RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK|RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_Clk.SYSCLKSource=RCC_SYSCLKSOURCE_PLLCLK;
  RCC_Clk.AHBCLKDivider=RCC_SYSCLK_DIV1;
  RCC_Clk.APB1CLKDivider=RCC_HCLK_DIV4;   // TIM2 clock = 2*PCLK1
  RCC_Clk.APB2CLKDivider=RCC_HCLK_DIV2;
  if (HAL_RCC_ClockConfig(&RCC_Clk, FLASH_LATENCY_5)!=HAL_OK) Error_Handler();
}

static void MX_GPIO_Init(void){
  GPIO_InitTypeDef GI={0};
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /* Saídas em nível baixo ao iniciar */
  HAL_GPIO_WritePin(GPIOB, FPGA_CS_Pin|LED_BLUE_Pin|MMA_AVDD_Pin, GPIO_PIN_RESET);  // PB5=MMA_AVDD
  HAL_GPIO_WritePin(GPIOA, LED_RED_Pin|LED_GREEN_Pin|LED_YELLOW_Pin, GPIO_PIN_RESET);

  /* PB: FPGA_CS, LED_BLUE, MMA_AVDD */
  GI.Pin = FPGA_CS_Pin|LED_BLUE_Pin|MMA_AVDD_Pin;
  GI.Mode=GPIO_MODE_OUTPUT_PP; GI.Pull=GPIO_NOPULL; GI.Speed=GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB,&GI);

  /* PA: LEDs */
  GI.Pin = LED_RED_Pin|LED_GREEN_Pin|LED_YELLOW_Pin;
  GI.Mode=GPIO_MODE_OUTPUT_PP; GI.Pull=GPIO_NOPULL; GI.Speed=GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA,&GI);

  /* Botão usuário (EXTI3) */
  GI.Pin = USER_Btn_Pin; GI.Mode=GPIO_MODE_IT_RISING; GI.Pull=GPIO_NOPULL;
  HAL_GPIO_Init(USER_Btn_GPIO_Port,&GI);
  HAL_NVIC_SetPriority(EXTI3_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI3_IRQn);
}

static void MX_DMA_Init(void){
  __HAL_RCC_DMA1_CLK_ENABLE();   // DAC (DMA1_Stream5)
  __HAL_RCC_DMA2_CLK_ENABLE();   // reservado p/ ADC1 (inativo)
  HAL_NVIC_SetPriority(DMA1_Stream5_IRQn, 7, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream5_IRQn);
}

static void MX_DAC_Init(void){
  DAC_ChannelConfTypeDef s={0};
  hdac.Instance = DAC;
  if (HAL_DAC_Init(&hdac)!=HAL_OK) Error_Handler();
  s.DAC_Trigger = DAC_TRIGGER_T2_TRGO;          // TIM2 -> TRGO UPDATE
  s.DAC_OutputBuffer = DAC_OUTPUTBUFFER_ENABLE;
  if (HAL_DAC_ConfigChannel(&hdac,&s,DAC_CHANNEL_1)!=HAL_OK) Error_Handler();
}

static void MX_I2C1_Init(void){
  hi2c1.Instance=I2C1;
  hi2c1.Init.ClockSpeed=100000;
  hi2c1.Init.DutyCycle=I2C_DUTYCYCLE_2;
  hi2c1.Init.OwnAddress1=0;
  hi2c1.Init.AddressingMode=I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode=I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2=0;
  hi2c1.Init.GeneralCallMode=I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode=I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1)!=HAL_OK) Error_Handler();
}

static void MX_SPI1_Init(void){
  hspi1.Instance=SPI1;
  hspi1.Init.Mode=SPI_MODE_MASTER;
  hspi1.Init.Direction=SPI_DIRECTION_2LINES;
  hspi1.Init.DataSize=SPI_DATASIZE_8BIT;
  hspi1.Init.CLKPolarity=SPI_POLARITY_LOW;
  hspi1.Init.CLKPhase=SPI_PHASE_1EDGE;
  hspi1.Init.NSS=SPI_NSS_SOFT;
  hspi1.Init.BaudRatePrescaler=SPI_BAUDRATEPRESCALER_8; // ~10.5 MHz @84MHz/8
  hspi1.Init.FirstBit=SPI_FIRSTBIT_MSB;
  hspi1.Init.TIMode=SPI_TIMODE_DISABLE;
  hspi1.Init.CRCCalculation=SPI_CRCCALCULATION_DISABLE;
  hspi1.Init.CRCPolynomial=10;
  if (HAL_SPI_Init(&hspi1)!=HAL_OK) Error_Handler();
  /* MOSI já está em PA7 no MSP */
}

static void MX_CRC_Init(void){
  hcrc.Instance=CRC; if (HAL_CRC_Init(&hcrc)!=HAL_OK) Error_Handler();
}

static void MX_TIM2_Init(void){
  TIM_ClockConfigTypeDef sClk={0};
  TIM_MasterConfigTypeDef sM={0};
  htim2.Instance=TIM2;
  htim2.Init.Prescaler=0;
  htim2.Init.CounterMode=TIM_COUNTERMODE_UP;
  htim2.Init.Period=4294967295;          // recalculado dinamicamente no dac_start()
  htim2.Init.ClockDivision=TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload=TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim2)!=HAL_OK) Error_Handler();
  sClk.ClockSource=TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim2,&sClk)!=HAL_OK) Error_Handler();
  sM.MasterOutputTrigger=TIM_TRGO_UPDATE;    // TRGO UPDATE -> DAC
  sM.MasterSlaveMode=TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2,&sM)!=HAL_OK) Error_Handler();
}

static void MX_TIM3_Init(void){
  // Mantido para compatibilidade, mas não usado sem ADC1.
  TIM_MasterConfigTypeDef sM={0};
  __HAL_RCC_TIM3_CLK_ENABLE();
  htim3.Instance=TIM3;
  htim3.Init.Prescaler=8400-1;          // 84MHz/8400 = 10kHz
  htim3.Init.CounterMode=TIM_COUNTERMODE_UP;
  htim3.Init.Period=100-1;              // 10kHz/100 = 100 Hz
  htim3.Init.ClockDivision=TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload=TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim3)!=HAL_OK) Error_Handler();
  sM.MasterOutputTrigger=TIM_TRGO_UPDATE;
  sM.MasterSlaveMode=TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3,&sM)!=HAL_OK) Error_Handler();
}

/* ============================ MAIN ============================ */
int main(void){
  HAL_Init();
  SystemClock_Config();

  MX_GPIO_Init();
  MX_DMA_Init();
  MX_DAC_Init();
  MX_I2C1_Init();
  MX_SPI1_Init();
  MX_CRC_Init();
  MX_TIM2_Init();
  MX_TIM3_Init();    // não usado sem ADC
  MX_USB_DEVICE_Init();

  printf("\r\nPyboard v1.1 — AWG/ACCEL/FPGA CLI pronto. fmax(sine,N=256)=%.1f Hz\r\n",
         fmax_from_fs(DAC_FS_MAX_HZ, LUT_N));
  printf("Digite HELP\r\n");
  put_prompt();

  uint32_t t_acc = HAL_GetTick();

  for(;;){
    /* 1) Comandos do USB CDC */
    if (cdc_cmd_ready){
      uint8_t local[128];
      uint32_t n=(cdc_cmd_length>sizeof(local)-1)?(sizeof(local)-1):cdc_cmd_length;
      for (uint32_t i=0;i<n;i++) local[i]=cdc_cmd_buffer[i];
      local[n]=0;
      cdc_cmd_ready=0;

      CLI_ProcessLine((const char*)local);
      put_prompt();
    }

    /* 2) Heartbeat */
    hb_tick();

    /* 3) Streaming do acelerômetro (~10 Hz) */
    if (g_app_state==STATE_STREAMING_ACCEL && HAL_GetTick()-t_acc>100){
      t_acc=HAL_GetTick();
      int8_t x,y,z;
      if (MMA7660_ReadXYZ(&hi2c1,&x,&y,&z)){
        float fx=x,fy=y,fz=z;
        if (g_kalman_on){ fx=kalman_step(&kx,fx); fy=kalman_step(&ky,fy); fz=kalman_step(&kz,fz); }
        printf("A:%d,%d,%d\r\n",(int)lrintf(fx),(int)lrintf(fy),(int)lrintf(fz));
      }
    }

    /* 4) Upload binário p/ FPGA */
    if (g_bin_mode) process_fpga_upload();
  }
}
