// Modelo de predicción de precios implementado en JavaScript puro
class PhonePricePredictor {
    constructor() {
        // Coeficientes del modelo (simulados)
        this.brandFactors = {
            'Apple': 1.8,
            'Samsung': 1.4,
            'Xiaomi': 1.1,
            'Huawei': 1.2,
            'Oppo': 1.0,
            'Vivo': 1.0,
            'Motorola': 1.0,
            'default': 1.0
        };
    }

    predict(features) {
        try {
            // Validar y normalizar entradas
            const brand = this.normalizeBrand(features.brand);
            const ram = parseInt(features.ram) || 4;
            const storage = parseInt(features.storage) || 64;
            const screen = parseFloat(features.screen) || 6.1;
            const camera = parseInt(features.camera) || 12;

            // Fórmula de predicción (modelo simplificado)
            const basePrice = 300;
            const brandFactor = this.brandFactors[brand] || this.brandFactors.default;
            
            let price = basePrice * brandFactor +
                       (ram * 25) +
                       (storage * 0.5) +
                       (screen * 40) +
                       (camera * 2);

            // Ajuste final y redondeo
            price = Math.max(100, Math.round(price * 0.9 + (Math.random() * price * 0.2)));

            return {
                success: true,
                price: price,
                features: {
                    brand: brand,
                    ram: ram,
                    storage: storage,
                    screen: screen,
                    camera: camera
                }
            };
        } catch (error) {
            return {
                success: false,
                error: "Error en la predicción: " + error.message
            };
        }
    }

    normalizeBrand(brand) {
        if (!brand) return 'default';
        
        const brandsMap = {
            'iphone': 'Apple',
            'galaxy': 'Samsung',
            'redmi': 'Xiaomi',
            'poco': 'Xiaomi'
        };

        const lowerBrand = brand.toLowerCase();
        for (const [key, value] of Object.entries(brandsMap)) {
            if (lowerBrand.includes(key)) {
                return value;
            }
        }

        // Capitalizar la primera letra si no está en el mapa
        return brand.charAt(0).toUpperCase() + brand.slice(1).toLowerCase();
    }

    extractFeatures(text) {
        const lowerText = text.toLowerCase();
        
        // Mejorado para capturar más variaciones
        const brandMatch = lowerText.match(/(iphone|galaxy|redmi|poco|samsung|apple|xiaomi|huawei|oppo|vivo|motorola|\b\w+\b)(?=\s|$)/) || ['unknown'];
        const ramMatch = lowerText.match(/(\d+)\s*(gb|gb ram|ram|memory)/) || [null, 4];
        const storageMatch = lowerText.match(/(\d+)\s*(gb|gb storage|storage|almacenamiento)/) || [null, 64];
        const screenMatch = lowerText.match(/(\d+\.?\d*)\s*(pulgadas|"|inch|pantalla)/) || [null, 6.1];
        const cameraMatch = lowerText.match(/(\d+)\s*(mp|mpx|cámara|camera|mega)/) || [null, 12];
        
        return {
            brand: brandMatch[0],
            ram: ramMatch[1],
            storage: storageMatch[1],
            screen: screenMatch[1],
            camera: cameraMatch[1]
        };
    }
}

// Modelo de predicción de precios para computadoras
class ComputerPriceModel {
    constructor() {
        // Factores de marca (precios base)
        this.brandFactors = {
            'Apple': 1.8,
            'Dell': 1.3,
            'HP': 1.2,
            'Lenovo': 1.1,
            'Asus': 1.1,
            'Acer': 1.0,
            'MSI': 1.4,
            'default': 1.0
        };
        
        // Factores para tipos de CPU
        this.cpuFactors = {
            'i3': 1.0,
            'i5': 1.3,
            'i7': 1.7,
            'i9': 2.2,
            'ryzen 5': 1.4,
            'ryzen 7': 1.8,
            'ryzen 9': 2.3,
            'm1': 1.9,
            'm2': 2.1,
            'default': 1.0
        };
        
        // Factores para GPU
        this.gpuFactors = {
            'integrada': 1.0,
            'mx': 1.2,
            'gtx': 1.5,
            'rtx 3050': 1.8,
            'rtx 3060': 2.0,
            'rtx 3070': 2.4,
            'rtx 3080': 2.8,
            'radeon': 1.7,
            'default': 1.0
        };
    }

    predict(features) {
        try {
            // Validar y normalizar entradas
            const brand = this.normalizeBrand(features.brand);
            const cpu = this.normalizeCPU(features.cpu);
            const gpu = this.normalizeGPU(features.gpu);
            const ram = parseInt(features.ram) || 8;
            const storage = parseInt(features.storage) || 256;
            const storageType = this.normalizeStorageType(features.storageType);
            const screen = parseFloat(features.screen) || 15.6;
            const os = features.os || 'Windows';

            // Fórmula de predicción mejorada
            const basePrice = 500;
            const brandFactor = this.brandFactors[brand] || this.brandFactors.default;
            const cpuFactor = this.cpuFactors[cpu] || this.cpuFactors.default;
            const gpuFactor = this.gpuFactors[gpu] || this.gpuFactors.default;
            let storageFactor = 1.0;
            
            if (storageType === 'ssd') storageFactor = 1.2;
            if (storageType === 'nvme') storageFactor = 1.4;
            
            let price = basePrice * brandFactor * cpuFactor * gpuFactor +
                       (ram * 25) +
                       (storage * 0.6 * storageFactor) +
                       (screen * 30);

            // Ajuste final y redondeo
            price = Math.max(300, Math.round(price));

            return {
                success: true,
                price: price,
                features: {
                    brand: brand,
                    cpu: cpu,
                    gpu: gpu,
                    ram: ram,
                    storage: storage,
                    storageType: storageType,
                    screen: screen,
                    os: os
                }
            };
        } catch (error) {
            return {
                success: false,
                error: "Error en la predicción: " + error.message
            };
        }
    }

    normalizeBrand(brand) {
        if (!brand) return 'default';
        
        const lowerBrand = brand.toLowerCase();
        const brandMappings = {
            'macbook': 'Apple',
            'mac': 'Apple',
            'alienware': 'Dell',
            'legion': 'Lenovo',
            'rog': 'Asus',
            'tuf': 'Asus'
        };

        for (const [key, value] of Object.entries(brandMappings)) {
            if (lowerBrand.includes(key)) {
                return value;
            }
        }

        return brand.charAt(0).toUpperCase() + brand.slice(1).toLowerCase();
    }

    normalizeCPU(cpu) {
        if (!cpu) return 'default';
        
        const lowerCPU = cpu.toLowerCase();
        const cpuMappings = {
            'core i3': 'i3',
            'core i5': 'i5',
            'core i7': 'i7',
            'core i9': 'i9',
            'intel i3': 'i3',
            'intel i5': 'i5',
            'intel i7': 'i7',
            'intel i9': 'i9',
            'ryzen': 'ryzen 5',
            'amd ryzen': 'ryzen 5'
        };

        for (const [key, value] of Object.entries(cpuMappings)) {
            if (lowerCPU.includes(key)) {
                return value;
            }
        }

        return lowerCPU;
    }

    normalizeGPU(gpu) {
        if (!gpu) return 'integrada';
        
        const lowerGPU = gpu.toLowerCase();
        const gpuMappings = {
            'nvidia': 'gtx',
            'geforce': 'gtx',
            'rtx': 'rtx 3060',
            'amd': 'radeon',
            'radeon': 'radeon',
            'iris xe': 'integrada',
            'uhd graphics': 'integrada'
        };

        for (const [key, value] of Object.entries(gpuMappings)) {
            if (lowerGPU.includes(key)) {
                return value;
            }
        }

        return lowerGPU;
    }

    normalizeStorageType(storageType) {
        if (!storageType) return 'hdd';
        
        const lowerType = storageType.toLowerCase();
        if (lowerType.includes('nvme')) return 'nvme';
        if (lowerType.includes('ssd')) return 'ssd';
        return 'hdd';
    }

    extractFeatures(text) {
        const lowerText = text.toLowerCase();
        
        // Expresiones regulares mejoradas
        const brandMatch = lowerText.match(/(macbook|mac|dell|hp|lenovo|asus|acer|msi|alienware|rog|\b\w+\b)(?=\s|$)/) || ['unknown'];
        const cpuMatch = lowerText.match(/(i3|i5|i7|i9|ryzen 3|ryzen 5|ryzen 7|ryzen 9|m1|m2|intel|amd|core i\d)/) || ['i5'];
        const gpuMatch = lowerText.match(/(rtx 3050|rtx 3060|rtx 3070|rtx 3080|gtx 1650|gtx 1660|mx\d+|integrada|radeon|nvidia|geforce)/) || ['integrada'];
        const ramMatch = lowerText.match(/(\d+)\s*(gb|gb ram|ram|memory)/) || [null, 8];
        const storageMatch = lowerText.match(/(\d+)\s*(gb|gb storage|storage|almacenamiento)/) || [null, 256];
        const storageTypeMatch = lowerText.match(/(ssd|hdd|nvme)/) || ['ssd'];
        const screenMatch = lowerText.match(/(\d+\.?\d*)\s*(pulgadas|"|inch|pantalla)/) || [null, 15.6];
        const osMatch = lowerText.match(/(windows|macos|linux|ubuntu|chrome os)/) || ['Windows'];
        
        return {
            brand: brandMatch[0],
            cpu: cpuMatch[0],
            gpu: gpuMatch[0],
            ram: ramMatch[1],
            storage: storageMatch[1],
            storageType: storageTypeMatch[0],
            screen: screenMatch[1],
            os: osMatch[0]
        };
    }
}

// Instancia global del modelo
const computerModel = new ComputerPriceModel();
const phonePredictor = new PhonePricePredictor();