#include "main.h"

/* Referências a handles definidos em main.c */
extern DAC_HandleTypeDef  hdac;
extern DMA_HandleTypeDef  hdma_dac1;

/* ========== Init global ========== */
void HAL_MspInit(void)
{
  __HAL_RCC_SYSCFG_CLK_ENABLE();
  __HAL_RCC_PWR_CLK_ENABLE();
}

/* ========== ADC ========== (deixe como está ou desabilite se quiser) */
void HAL_ADC_MspInit(ADC_HandleTypeDef* hadc)
{
  if(hadc->Instance==ADC1)
  {
    __HAL_RCC_ADC1_CLK_ENABLE();
    /* Se for usar ADC1 + DMA depois, configure aqui o GPIO do canal e o DMA2_Stream0 */
  }
  else if(hadc->Instance==ADC2)
  {
    __HAL_RCC_ADC2_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();
    GPIO_InitTypeDef GI = {0};
    GI.Pin = GPIO_PIN_1; GI.Mode = GPIO_MODE_ANALOG; GI.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOA, &GI);
  }
  else if(hadc->Instance==ADC3)
  {
    __HAL_RCC_ADC3_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();
    GPIO_InitTypeDef GI = {0};
    GI.Pin = GPIO_PIN_3; GI.Mode = GPIO_MODE_ANALOG; GI.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOA, &GI);
  }
}

void HAL_ADC_MspDeInit(ADC_HandleTypeDef* hadc)
{
  if(hadc->Instance==ADC1) { __HAL_RCC_ADC1_CLK_DISABLE(); }
  else if(hadc->Instance==ADC2){ __HAL_RCC_ADC2_CLK_DISABLE(); HAL_GPIO_DeInit(GPIOA, GPIO_PIN_1); }
  else if(hadc->Instance==ADC3){ __HAL_RCC_ADC3_CLK_DISABLE(); HAL_GPIO_DeInit(GPIOA, GPIO_PIN_3); }
}

/* ========== CRC ========== */
void HAL_CRC_MspInit(CRC_HandleTypeDef* hcrc)
{
  if(hcrc->Instance==CRC) { __HAL_RCC_CRC_CLK_ENABLE(); }
}
void HAL_CRC_MspDeInit(CRC_HandleTypeDef* hcrc)
{
  if(hcrc->Instance==CRC) { __HAL_RCC_CRC_CLK_DISABLE(); }
}

/* ========== DAC + DMA ==========
   DAC1_CH1 usa DMA1 Stream5 Channel7 no F405  */
void HAL_DAC_MspInit(DAC_HandleTypeDef* hdac)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if(hdac->Instance==DAC)
  {
    __HAL_RCC_DAC_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_DMA1_CLK_ENABLE();

    /* PA4 -> DAC_OUT1 */
    GPIO_InitStruct.Pin = GPIO_PIN_4;
    GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    /* DMA1 Stream5 Channel7 -> DAC_CH1, mem->periph, circular */
    hdma_dac1.Instance = DMA1_Stream5;
    hdma_dac1.Init.Channel = DMA_CHANNEL_7;
    hdma_dac1.Init.Direction = DMA_MEMORY_TO_PERIPH;
    hdma_dac1.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_dac1.Init.MemInc = DMA_MINC_ENABLE;
    hdma_dac1.Init.PeriphDataAlignment = DMA_PDATAALIGN_HALFWORD;
    hdma_dac1.Init.MemDataAlignment = DMA_MDATAALIGN_HALFWORD;
    hdma_dac1.Init.Mode = DMA_CIRCULAR;
    hdma_dac1.Init.Priority = DMA_PRIORITY_HIGH;
    hdma_dac1.Init.FIFOMode = DMA_FIFOMODE_DISABLE;
    if (HAL_DMA_Init(&hdma_dac1) != HAL_OK) { Error_Handler(); }

    __HAL_LINKDMA(hdac, DMA_Handle1, hdma_dac1);

    /* NVIC do DMA (se ainda não tiver) */
    HAL_NVIC_SetPriority(DMA1_Stream5_IRQn, 7, 0);
    HAL_NVIC_EnableIRQ(DMA1_Stream5_IRQn);
  }
}

void HAL_DAC_MspDeInit(DAC_HandleTypeDef* hdac_)
{
  if(hdac_->Instance==DAC)
  {
    __HAL_RCC_DAC_CLK_DISABLE();
    HAL_GPIO_DeInit(GPIOA, GPIO_PIN_4);
    HAL_DMA_DeInit(hdac_->DMA_Handle1);
  }
}

/* ========== I2C1 ========== */
void HAL_I2C_MspInit(I2C_HandleTypeDef* hi2c)
{
  if(hi2c->Instance==I2C1)
  {
    __HAL_RCC_GPIOB_CLK_ENABLE();
    GPIO_InitTypeDef GI = {0};
    GI.Pin = GPIO_PIN_6|GPIO_PIN_7;
    GI.Mode = GPIO_MODE_AF_OD;
    GI.Pull = GPIO_NOPULL;
    GI.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GI.Alternate = GPIO_AF4_I2C1;
    HAL_GPIO_Init(GPIOB, &GI);

    __HAL_RCC_I2C1_CLK_ENABLE();
  }
}
void HAL_I2C_MspDeInit(I2C_HandleTypeDef* hi2c)
{
  if(hi2c->Instance==I2C1)
  {
    __HAL_RCC_I2C1_CLK_DISABLE();
    HAL_GPIO_DeInit(GPIOB, GPIO_PIN_6|GPIO_PIN_7);
  }
}

/* ========== SPI1 ==========
   TROCA: MOSI agora é **PA7** (não mais PB5)  */
void HAL_SPI_MspInit(SPI_HandleTypeDef* hspi)
{
  if(hspi->Instance==SPI1)
  {
    __HAL_RCC_SPI1_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();

    GPIO_InitTypeDef GI = {0};
    /* PA5=SCK, PA6=MISO, PA7=MOSI */
    GI.Pin = GPIO_PIN_5 | GPIO_PIN_6 | GPIO_PIN_7;
    GI.Mode = GPIO_MODE_AF_PP;
    GI.Pull = GPIO_NOPULL;
    GI.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GI.Alternate = GPIO_AF5_SPI1;
    HAL_GPIO_Init(GPIOA, &GI);
  }
}

void HAL_SPI_MspDeInit(SPI_HandleTypeDef* hspi)
{
  if(hspi->Instance==SPI1)
  {
    __HAL_RCC_SPI1_CLK_DISABLE();
    HAL_GPIO_DeInit(GPIOA, GPIO_PIN_5|GPIO_PIN_6|GPIO_PIN_7);
  }
}

/* ========== TIM2 (gerador de trigger do DAC) ========== */
void HAL_TIM_Base_MspInit(TIM_HandleTypeDef* htim_base)
{
  if(htim_base->Instance==TIM2)
  {
    __HAL_RCC_TIM2_CLK_ENABLE();
    /* Não precisamos habilitar IRQ do TIM2 para o DAC funcionar em TRGO */
  }
}

void HAL_TIM_Base_MspDeInit(TIM_HandleTypeDef* htim_base)
{
  if(htim_base->Instance==TIM2)
  {
    __HAL_RCC_TIM2_CLK_DISABLE();
  }
}
