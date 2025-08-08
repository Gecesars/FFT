#ifndef ACCEL_MMA7660_H
#define ACCEL_MMA7660_H

#include "stm32f4xx_hal.h"
#include <stdbool.h>
#include <stdint.h>

/* Endereço I2C 7 bits do MMA7660 (0x4C). HAL usa 8-bit address -> shift <<1 */
#define MMA7660_I2C_ADDR            (0x4C << 1)

/* Registradores principais */
#define MMA7660_REG_X_OUT           0x00
#define MMA7660_REG_Y_OUT           0x01
#define MMA7660_REG_Z_OUT           0x02
#define MMA7660_REG_TILT            0x03
#define MMA7660_REG_MODE            0x07
#define MMA7660_REG_SR              0x08

/* Bits do MODE */
#define MMA7660_MODE_ACTIVE         0x01
#define MMA7660_MODE_STANDBY        0x00

/* Exemplos de Sample Rate (datasheet) – usaremos 64 Hz */
#define MMA7660_SR_120HZ            0x00
#define MMA7660_SR_64HZ             0x01
#define MMA7660_SR_32HZ             0x02
#define MMA7660_SR_16HZ             0x03
/* ... (outros valores possíveis) ... */

/* Converte 6 bits 2’s-complement em int8_t (−32..+31) */
#define MMA7660_AXIS_SIGNED_VALUE(i)  ( (int8_t)(((i) & 0x3F) | (((i) & 0x20) ? 0xC0 : 0x00)) )

#ifdef __cplusplus
extern "C" {
#endif

/** Inicializa o MMA7660.
 *  - Liga AVDD (gpio_avdd/gpio_pin)
 *  - Verifica presença no I2C
 *  - MODE=STANDBY -> SR=64Hz -> MODE=ACTIVE
 */
bool MMA7660_Init(I2C_HandleTypeDef *hi2c,
                  GPIO_TypeDef *gpio_avdd, uint16_t gpio_pin);

/** Lê um eixo bruto (valor −32..+31) */
bool MMA7660_ReadAxis(I2C_HandleTypeDef *hi2c, uint8_t axis_reg, int8_t *value);

/** Lê X, Y, Z (cada um −32..+31) */
bool MMA7660_ReadXYZ(I2C_HandleTypeDef *hi2c, int8_t *x, int8_t *y, int8_t *z);

#ifdef __cplusplus
}
#endif

#endif /* ACCEL_MMA7660_H */
