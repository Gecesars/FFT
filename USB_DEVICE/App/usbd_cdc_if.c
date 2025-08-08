/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : usbd_cdc_if.c
  * @version        : v1.0_Cube
  * @brief          : USB CDC (Virtual COM) - implementação robusta
  *
  * Correções principais:
  *   - CDC_Receive_FS: caminho "modo texto" monta linhas em cdc_cmd_buffer,
  *     seta cdc_cmd_ready e re-arma o EP (SetRxBuffer + ReceivePacket).
  *   - CDC_Receive_FS: caminho "modo binário" repassa ao hook CDC_OnRxData().
  *   - _write(): envia por VCP com timeout, sem travar em contexto de IRQ.
  *   - Evita qualquer printf dentro de IRQ (isso é tratado no seu código da app).
  ******************************************************************************
  * @attention
  * Copyright (c) 2025.
  * Provided AS-IS.
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "usbd_cdc_if.h"

/* USER CODE BEGIN Includes */
#include "main.h"
#include <string.h>
#include <stdbool.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define RX_LINE_BUF_SZ   128u
#define TX_TIMEOUT_MS    25u
/* Private macro -------------------------------------------------------------*/

/* USER CODE BEGIN Extern hooks */
extern uint8_t App_IsBinaryMode(void);      /* em main.c */
extern void    CDC_OnRxData(uint8_t *buf, uint32_t len); /* em main.c */
/* Variáveis exportadas (consumidas no superloop em main.c) */
volatile uint8_t  cdc_cmd_buffer[RX_LINE_BUF_SZ];
volatile uint32_t cdc_cmd_length = 0;
volatile uint8_t  cdc_cmd_ready  = 0;
/* USER CODE END Extern hooks */

/* Private variables ---------------------------------------------------------*/
extern USBD_HandleTypeDef hUsbDeviceFS;

/* Create buffer for reception and transmission           */
__ALIGN_BEGIN uint8_t UserRxBufferFS[APP_RX_DATA_SIZE] __ALIGN_END;
__ALIGN_BEGIN uint8_t UserTxBufferFS[APP_TX_DATA_SIZE] __ALIGN_END;

/* USER CODE BEGIN PV */
static USBD_CDC_HandleTypeDef *pCDC = NULL;      /* acesso rápido a TxState */
static uint8_t s_linebuf[RX_LINE_BUF_SZ];        /* montagem de linha */
static uint32_t s_linepos = 0;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
static int8_t CDC_Init_FS(void);
static int8_t CDC_DeInit_FS(void);
static int8_t CDC_Control_FS(uint8_t cmd, uint8_t* pbuf, uint16_t length);
static int8_t CDC_Receive_FS(uint8_t* pbuf, uint32_t *Len);
static int8_t CDC_TransmitCplt_FS(uint8_t *pbuf, uint32_t *Len, uint8_t epnum);

/* USB handler structure -----------------------------------------------------*/
USBD_CDC_ItfTypeDef USBD_Interface_fops_FS =
{
  CDC_Init_FS,
  CDC_DeInit_FS,
  CDC_Control_FS,
  CDC_Receive_FS,
  CDC_TransmitCplt_FS
};

/* Functions -----------------------------------------------------------------*/
/**
  * @brief  Initializes the CDC media low layer
  */
static int8_t CDC_Init_FS(void)
{
  USBD_CDC_SetTxBuffer(&hUsbDeviceFS, UserTxBufferFS, 0);
  USBD_CDC_SetRxBuffer(&hUsbDeviceFS, UserRxBufferFS);

  pCDC = (USBD_CDC_HandleTypeDef*)hUsbDeviceFS.pClassData;
  s_linepos = 0;
  return (USBD_OK);
}

/**
  * @brief  DeInitializes the CDC media low layer
  */
static int8_t CDC_DeInit_FS(void)
{
  pCDC = NULL;
  return (USBD_OK);
}

/**
  * @brief  Manage the CDC class requests
  */
static int8_t CDC_Control_FS(uint8_t cmd, uint8_t* pbuf, uint16_t length)
{
  (void)cmd; (void)pbuf; (void)length;
  return (USBD_OK);
}

/**
  * @brief  Data received over USB OUT endpoint.
  *         Se a aplicação estiver em "modo binário", os bytes são
  *         encaminhados crus ao hook CDC_OnRxData(). Caso contrário,
  *         montamos uma linha até '\n' ou '\r' e sinalizamos o parser.
  */
static int8_t CDC_Receive_FS(uint8_t* Buf, uint32_t *Len)
{
  if (*Len == 0) goto rearm;

  if (App_IsBinaryMode())
  {
    CDC_OnRxData(Buf, *Len);
  }
  else
  {
    for (uint32_t i = 0; i < *Len; i++)
    {
      uint8_t c = Buf[i];
      if (c == '\n' || c == '\r')
      {
        if (s_linepos > 0 && !cdc_cmd_ready)
        {
          uint32_t n = (s_linepos >= RX_LINE_BUF_SZ-1) ? (RX_LINE_BUF_SZ-1) : s_linepos;
          for (uint32_t k=0; k<n; k++) cdc_cmd_buffer[k] = s_linebuf[k];
          cdc_cmd_buffer[n] = 0;
          cdc_cmd_length = n;
          cdc_cmd_ready  = 1;
        }
        s_linepos = 0;
      }
      else
      {
        if (s_linepos < RX_LINE_BUF_SZ-1)
          s_linebuf[s_linepos++] = c;
      }
    }
  }

rearm:
  USBD_CDC_SetRxBuffer(&hUsbDeviceFS, &Buf[0]);
  USBD_CDC_ReceivePacket(&hUsbDeviceFS);
  return (USBD_OK);
}

/**
  * @brief  Tx complete callback
  */
static int8_t CDC_TransmitCplt_FS(uint8_t *Buf, uint32_t *Len, uint8_t epnum)
{
  (void)Buf; (void)Len; (void)epnum;
  return (USBD_OK);
}

/* ===================== Retarget printf ==================================== */
/* Envia payload por CDC com timeout (não travar em IRQ nem loop infinito).   */
int _write(int file, char *ptr, int len)
{
  (void)file;

  if (len <= 0 || hUsbDeviceFS.pClassData == NULL)
    return 0;

  /* NUNCA transmitir de contexto de interrupção */
  if (SCB->ICSR & SCB_ICSR_VECTACTIVE_Msk)
    return len; /* descarta silenciosamente em IRQ */

  uint32_t t0 = HAL_GetTick();
  USBD_CDC_HandleTypeDef *hcdc = (USBD_CDC_HandleTypeDef*)hUsbDeviceFS.pClassData;

  while (hcdc->TxState != 0)
  {
    if ((HAL_GetTick() - t0) > TX_TIMEOUT_MS) return 0; /* timeout */
  }

  USBD_CDC_SetTxBuffer(&hUsbDeviceFS, (uint8_t*)ptr, (uint16_t)len);
  if (USBD_CDC_TransmitPacket(&hUsbDeviceFS) != USBD_OK) return 0;

  /* aguarda a DMA/USB pegar o buffer (TxState zerar) com timeout curto */
  t0 = HAL_GetTick();
  while (hcdc->TxState != 0)
  {
    if ((HAL_GetTick() - t0) > TX_TIMEOUT_MS) break;
  }
  return len;
}
/* ========================================================================== */
