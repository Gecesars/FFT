/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Cabeçalho principal da aplicação
  *
  * Principais decisões:
  *  - Inclui aliases para LED_RED_* e MMA_AVDD_* se não existirem no projeto.
  *  - Define AppState (estado da aplicação) de forma única para todos os módulos.
  *  - Exporta protótipos e externs úteis.
  ******************************************************************************
  */
/* USER CODE END Header */

#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal.h"

/* ======== Pinos gerados pelo Cube (ajuste conforme o seu .ioc) ======== */
#define FPGA_CS_Pin           GPIO_PIN_10
#define FPGA_CS_GPIO_Port     GPIOB

#define LED_GREEN_Pin         GPIO_PIN_14
#define LED_GREEN_GPIO_Port   GPIOA
#define LED_YELLOW_Pin        GPIO_PIN_15
#define LED_YELLOW_GPIO_Port  GPIOA

#define LED_BLUE_Pin          GPIO_PIN_4
#define LED_BLUE_GPIO_Port    GPIOB

#define USER_Btn_Pin          GPIO_PIN_3
#define USER_Btn_GPIO_Port    GPIOB
#define USER_Btn_EXTI_IRQn    EXTI3_IRQn

/* ======== Aliases para compatibilidade com o firmware (se faltarem) ======== */
#ifndef LED_RED_Pin
#define LED_RED_Pin           GPIO_PIN_13
#define LED_RED_GPIO_Port     GPIOA
#endif

#ifndef MMA_AVDD_Pin
/* AVDD do MMA7660 — ajuste ao seu hardware: PB5 é comum na Pyboard v1.x */
#define MMA_AVDD_Pin          GPIO_PIN_5
#define MMA_AVDD_GPIO_Port    GPIOB
#endif

/* ======== Tipos globais ======== */
typedef enum {
  STATE_IDLE = 0,
  STATE_STREAMING_ACCEL = 1
} AppState;

/* ======== Externs úteis ======== */
void Error_Handler(void);

/* Buffers de comando (produzidos em usbd_cdc_if.c, consumidos no main.c) */
extern volatile uint8_t  cdc_cmd_buffer[128];
extern volatile uint32_t cdc_cmd_length;
extern volatile uint8_t  cdc_cmd_ready;

#ifdef __cplusplus
}
#endif
#endif /* __MAIN_H */
