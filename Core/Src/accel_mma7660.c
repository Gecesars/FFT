#include "accel_mma7660.h"

static bool write_reg(I2C_HandleTypeDef *hi2c, uint8_t reg, uint8_t val) {
    for (int t = 0; t < 3; ++t) {
        if (HAL_I2C_Mem_Write(hi2c, MMA7660_I2C_ADDR, reg, I2C_MEMADD_SIZE_8BIT,
                              &val, 1, 100) == HAL_OK) {
            return true;
        }
        HAL_Delay(2);
    }
    return false;
}

static bool read_reg(I2C_HandleTypeDef *hi2c, uint8_t reg, uint8_t *val) {
    for (int t = 0; t < 3; ++t) {
        if (HAL_I2C_Mem_Read(hi2c, MMA7660_I2C_ADDR, reg, I2C_MEMADD_SIZE_8BIT,
                             val, 1, 100) == HAL_OK) {
            return true;
        }
        HAL_Delay(2);
    }
    return false;
}

bool MMA7660_Init(I2C_HandleTypeDef *hi2c,
                  GPIO_TypeDef *gpio_avdd, uint16_t gpio_pin)
{
    /* Alimentação AVDD via PB5 (conforme esquemático) */
    HAL_GPIO_WritePin(gpio_avdd, gpio_pin, GPIO_PIN_RESET);
    HAL_Delay(30);
    HAL_GPIO_WritePin(gpio_avdd, gpio_pin, GPIO_PIN_SET);
    HAL_Delay(30);

    /* Presença no barramento (até 4 tentativas) */
    for (int i = 0; i < 4; ++i) {
        if (HAL_I2C_IsDeviceReady(hi2c, MMA7660_I2C_ADDR, 1, 100) == HAL_OK) {
            break;
        }
        if (i == 3) {
            return false; /* não respondeu */
        }
        HAL_Delay(5);
    }

    /* Entrar em STANDBY para configurar */
    if (!write_reg(hi2c, MMA7660_REG_MODE, MMA7660_MODE_STANDBY)) {
        return false;
    }

    /* Sample Rate = 64 Hz (qualquer outro pode ser usado) */
    if (!write_reg(hi2c, MMA7660_REG_SR, MMA7660_SR_64HZ)) {
        return false;
    }

    /* Ativar */
    if (!write_reg(hi2c, MMA7660_REG_MODE, MMA7660_MODE_ACTIVE)) {
        return false;
    }

    HAL_Delay(10);
    return true;
}

bool MMA7660_ReadAxis(I2C_HandleTypeDef *hi2c, uint8_t axis_reg, int8_t *value)
{
    uint8_t raw;
    if (!read_reg(hi2c, axis_reg, &raw)) {
        return false;
    }
    *value = MMA7660_AXIS_SIGNED_VALUE(raw);
    return true;
}

bool MMA7660_ReadXYZ(I2C_HandleTypeDef *hi2c, int8_t *x, int8_t *y, int8_t *z)
{
    uint8_t regs[3] = { MMA7660_REG_X_OUT, MMA7660_REG_Y_OUT, MMA7660_REG_Z_OUT };
    uint8_t raw;
    if (!read_reg(hi2c, regs[0], &raw)) return false; *x = MMA7660_AXIS_SIGNED_VALUE(raw);
    if (!read_reg(hi2c, regs[1], &raw)) return false; *y = MMA7660_AXIS_SIGNED_VALUE(raw);
    if (!read_reg(hi2c, regs[2], &raw)) return false; *z = MMA7660_AXIS_SIGNED_VALUE(raw);
    return true;
}
