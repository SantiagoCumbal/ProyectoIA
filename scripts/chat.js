// Variables globales
let currentMode = 'phones';
let isLoading = false;
const messageInput = document.getElementById('messageInput');
const chatMessages = document.getElementById('chatMessages');

// Función para añadir mensajes al chat
function addMessage(content, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    const avatar = document.createElement('div');
    avatar.className = `message-avatar ${sender}-avatar`;
    avatar.textContent = sender === 'user' ? 'Tú' : 'AI';
    
    const text = document.createElement('div');
    text.className = 'message-text';
    
    if (typeof content === 'string' && content.startsWith('<')) {
        text.innerHTML = content;
    } else {
        text.textContent = content;
    }
    
    messageContent.appendChild(avatar);
    messageContent.appendChild(text);
    messageDiv.appendChild(messageContent);
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Función para mostrar el indicador de "escribiendo"
function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.id = 'typingIndicator';
    
    typingDiv.innerHTML = `
        <div class="message-content">
            <div class="message-avatar bot-avatar">AI</div>
            <div class="message-text typing-dots">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Función para ocultar el indicador de "escribiendo"
function hideTypingIndicator() {
    const typing = document.getElementById('typingIndicator');
    if (typing) typing.remove();
}

// Función para formatear resultados
function formatPredictionResult(prediction) {
    const isComputer = currentMode === 'computers';
    const deviceType = isComputer ? 'computadora' : 'celular';
    const deviceIcon = isComputer ? '💻' : '📱';
    
    let featuresHtml = '';
    
    if (isComputer) {
        featuresHtml = `
            <div class="feature-item">• Marca: ${prediction.features.brand}</div>
            <div class="feature-item">• Procesador: ${prediction.features.cpu.toUpperCase()}</div>
            <div class="feature-item">• RAM: ${prediction.features.ram} GB</div>
            <div class="feature-item">• GPU: ${prediction.features.gpu.toUpperCase()}</div>
            <div class="feature-item">• Almacenamiento: ${prediction.features.storage} GB ${prediction.features.storageType.toUpperCase()}</div>
            <div class="feature-item">• Pantalla: ${prediction.features.screen}"</div>
            <div class="feature-item">• Sistema: ${prediction.features.os}</div>
        `;
    } else {
        featuresHtml = `
            <div class="feature-item">• Marca: ${prediction.features.brand}</div>
            <div class="feature-item">• RAM: ${prediction.features.ram} GB</div>
            <div class="feature-item">• Almacenamiento: ${prediction.features.storage} GB</div>
            <div class="feature-item">• Pantalla: ${prediction.features.screen}"</div>
            ${prediction.features.camera ? `<div class="feature-item">• Cámara: ${prediction.features.camera} MP</div>` : ''}
        `;
    }
    
    return `
        <div class="prediction-result ${isComputer ? 'computer' : 'phone'}">
            <div class="prediction-title">${deviceIcon} Precio estimado para ${deviceType}</div>
            <div class="prediction-value">$${prediction.price.toLocaleString()} USD</div>
            <div class="features-list">
                <div class="prediction-title">📝 Especificaciones detectadas:</div>
                ${featuresHtml}
            </div>
            <div class="disclaimer">
                ${deviceIcon} Precio de referencia basado en características técnicas del mercado
            </div>
        </div>
    `;
}

// Función para enviar mensajes
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isLoading) return;

    // Limpiar input
    messageInput.value = '';
    messageInput.style.height = 'auto';

    // Mostrar mensaje del usuario
    addMessage(message, 'user');

    // Mostrar indicador de escritura
    showTypingIndicator();
    isLoading = true;

    try {
        let prediction;
        let features;
        
        if (currentMode === 'phones') {
            features = phoneModel.extractFeatures(message);
            prediction = phoneModel.predict(features);
        } else {
            features = computerModel.extractFeatures(message);
            prediction = computerModel.predict(features);
        }

        if (!prediction.success) {
            addMessage(prediction.error, 'bot');
            return;
        }

        // Mostrar resultados formateados
        const resultHtml = formatPredictionResult(prediction);
        addMessage(resultHtml, 'bot');

    } catch (error) {
        addMessage("⚠️ No pude analizar las especificaciones. Intenta con un formato como: " + 
                  (currentMode === 'phones' ? 
                   "'Marca, RAM, almacenamiento, pantalla, cámara'" : 
                   "'Marca, CPU, RAM, GPU, almacenamiento, pantalla'"), 'bot');
    } finally {
        hideTypingIndicator();
        isLoading = false;
    }
}

// Función para cambiar entre modos
function switchMode(mode) {
    if (mode === currentMode || isLoading) return;
    
    currentMode = mode;
    
    // Actualizar botones
    document.querySelectorAll('.mode-button').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`${mode}Mode`).classList.add('active');
    
    // Actualizar título e indicador
    const indicatorEl = document.getElementById('modeIndicator');
    const welcomeMessages = {
        'phones': "¡Hola! Soy tu asistente de precios de celulares. Puedes preguntarme cosas como:<br><br>" +
                 "• <em>'Cuánto cuesta un iPhone con 8GB RAM y 256GB?'</em><br>" +
                 "• <em>'Precio de Samsung Galaxy 128GB 6.5 pulgadas'</em>",
        'computers': "¡Hola! Soy tu asistente de precios de computadoras. Ejemplos de consultas:<br><br>" +
                    "• <em>'Laptop Dell i7 16GB RTX 3060 512GB SSD'</em><br>" +
                    "• <em>'PC Gamer Ryzen 7 32GB RTX 3070 1TB NVMe'</em>"
    };
    
    if (mode === 'phones') {
        indicatorEl.textContent = 'Celulares';
        indicatorEl.className = 'mode-indicator phones-indicator';
        messageInput.placeholder = "Ej: Samsung Galaxy S23, 8GB RAM, 128GB, 6.1\", 50MP";
    } else {
        indicatorEl.textContent = 'Computadoras';
        indicatorEl.className = 'mode-indicator computers-indicator';
        messageInput.placeholder = "Ej: Laptop Dell, Intel i7, 16GB RAM, RTX 3060, 512GB SSD, 15.6\"";
    }
    
    // Mostrar mensaje de bienvenida del modo actual
    chatMessages.innerHTML = '';
    addMessage(welcomeMessages[mode], 'bot');
}

// Función para nueva conversación
function newChat() {
    chatMessages.innerHTML = '';
    const welcomeMessage = currentMode === 'phones' ?
        "¡Hola! Soy tu asistente de precios de celulares. ¿En qué puedo ayudarte hoy?" :
        "¡Hola! Soy tu asistente de precios de computadoras. ¿Qué configuración necesitas evaluar?";
    addMessage(welcomeMessage, 'bot');
}

// Event listeners
messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

messageInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Inicialización
document.addEventListener('DOMContentLoaded', function() {
    messageInput.focus();
    
    // Ejemplo de historial de conversación
    const historyList = document.getElementById('chatHistoryList');
    const examples = [
        {text: "iPhone 13 6GB RAM 128GB", mode: "phones"},
        {text: "Samsung Galaxy 8GB 256GB", mode: "phones"},
        {text: "Laptop Dell i7 16GB RAM RTX 3060", mode: "computers"},
        {text: "MacBook Pro M1 16GB 512GB", mode: "computers"}
    ];
    
    examples.forEach(example => {
        const item = document.createElement('div');
        item.className = 'chat-item';
        item.textContent = example.text;
        item.onclick = () => {
            switchMode(example.mode);
            messageInput.value = example.text;
            messageInput.focus();
        };
        historyList.appendChild(item);
    });
    
    // Mensaje de bienvenida inicial
    addMessage("¡Hola! Soy tu asistente de precios de celulares. Puedes preguntarme cosas como:<br><br>" +
               "• <em>'Cuánto cuesta un iPhone con 8GB RAM y 256GB?'</em><br>" +
               "• <em>'Precio de Samsung Galaxy 128GB 6.5 pulgadas'</em>", 'bot');
});