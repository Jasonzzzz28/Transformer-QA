// DOM Elements
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const chatMessages = document.getElementById('chat-messages');
const clearChatBtn = document.getElementById('clear-chat');
const settingsBtn = document.getElementById('settings-btn');
const settingsModal = document.getElementById('settings-modal');
const closeSettings = document.getElementById('close-settings');
const saveSettings = document.getElementById('save-settings');
const modelStatus = document.getElementById('model-status');

function insertSampleQuestion(element) {
    userInput.value = element.textContent;
    userInput.focus();
}

function addMessage(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', 'fade-in');
    
    if (isUser) {
        messageDiv.classList.add('ml-auto', 'bg-gray-100', 'p-4', 'rounded-lg', 'rounded-tr-none');
        messageDiv.innerHTML = `
            <div class="flex items-start justify-end">
                <div>
                    <p class="font-semibold text-gray-700 text-right">You</p>
                    <p class="text-gray-700 mt-1">${content}</p>
                </div>
                <div class="flex-shrink-0 h-8 w-8 rounded-full bg-gray-200 flex items-center justify-center ml-3">
                    <i class="fas fa-user text-gray-600"></i>
                </div>
            </div>
        `;
    } else {
        messageDiv.classList.add('bg-blue-50', 'p-4', 'rounded-lg', 'rounded-tl-none');
        messageDiv.innerHTML = `
            <div class="flex items-start">
                <div class="flex-shrink-0 h-8 w-8 rounded-full bg-blue-100 flex items-center justify-center mr-3">
                    <i class="fas fa-robot text-blue-500"></i>
                </div>
                <div>
                    <p class="font-semibold text-blue-700">Transformers Bot</p>
                    <p class="text-gray-700 mt-1">${content}</p>
                </div>
            </div>
        `;
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.classList.add('message', 'bg-blue-50', 'p-4', 'rounded-lg', 'rounded-tl-none');
    typingDiv.innerHTML = `
        <div class="flex items-start">
            <div class="flex-shrink-0 h-8 w-8 rounded-full bg-blue-100 flex items-center justify-center mr-3">
                <i class="fas fa-robot text-blue-500"></i>
            </div>
            <div>
                <p class="font-semibold text-blue-700">Transformers Bot</p>
                <p class="text-gray-700 mt-1 typing-indicator">Thinking</p>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return typingDiv;
}

function removeTypingIndicator(typingDiv) {
    if (typingDiv && typingDiv.parentNode) {
        typingDiv.parentNode.removeChild(typingDiv);
    }
}

async function getBotResponse(question) {
    // Show loading state
    modelStatus.textContent = "Model processing...";
    modelStatus.classList.add('text-yellow-600');
    modelStatus.classList.remove('text-blue-600', 'text-red-600');
    
    const typingDiv = showTypingIndicator();
    
    try {
        // Call Flask API
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question }),
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        removeTypingIndicator(typingDiv);
        addMessage(data.response);
        
        // Update status
        modelStatus.textContent = "Model ready";
        modelStatus.classList.add('text-blue-600');
        modelStatus.classList.remove('text-yellow-600', 'text-red-600');
        
        return data.response;
    } catch (error) {
        removeTypingIndicator(typingDiv);
        addMessage("Sorry, I encountered an error processing your question. Please try again.");
        
        // Update status
        modelStatus.textContent = "Error - try again";
        modelStatus.classList.add('text-red-600');
        modelStatus.classList.remove('text-blue-600', 'text-yellow-600');
        
        console.error('Error:', error);
    }
}

// Event Listeners
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = userInput.value.trim();
    
    if (question) {
        addMessage(question, true);
        userInput.value = '';
        await getBotResponse(question);
    }
});

clearChatBtn.addEventListener('click', () => {
    chatMessages.innerHTML = '';
    addMessage("Hello! I'm a QA chatbot powered by Hugging Face Transformers. Ask me anything about the Transformers library, NLP models, or AI in general!");
});

settingsBtn.addEventListener('click', () => {
    settingsModal.classList.remove('hidden');
});

closeSettings.addEventListener('click', () => {
    settingsModal.classList.add('hidden');
});

saveSettings.addEventListener('click', () => {
    settingsModal.classList.add('hidden');
    // TODO: Save these settings and update the model
    addMessage("Settings updated! My behavior may change based on your new preferences.");
});

// Close modal when clicking outside
settingsModal.addEventListener('click', (e) => {
    if (e.target === settingsModal) {
        settingsModal.classList.add('hidden');
    }
});

// Initialize chat
userInput.focus();