#include "main.h"
#include <stdio.h>

void Error_Handler(void) {
    __disable_irq();
    while(1) {
        // Indicar erro com LEDs
        HAL_GPIO_TogglePin(LED_RED_GPIO_Port, LED_RED_Pin);
        HAL_GPIO_TogglePin(LED_GREEN_GPIO_Port, LED_GREEN_Pin);
        HAL_GPIO_TogglePin(LED_YELLOW_GPIO_Port, LED_YELLOW_Pin);
        HAL_GPIO_TogglePin(LED_BLUE_GPIO_Port, LED_BLUE_Pin);
        HAL_Delay(500);
    }
}
