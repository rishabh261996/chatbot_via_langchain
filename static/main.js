function sendMessage() {
    let messageInput = document.getElementById('message-input');
    let message = messageInput.value;
    displayMessage('user', message);

    // Get the selected function from the dropdown menu
    let functionSelect = document.getElementById('function-select');
    let selectedFunction = functionSelect.value;

    // Send an AJAX request to the Flask API endpoint based on the selected function
    let xhr = new XMLHttpRequest();
    let url;

    switch (selectedFunction) {
        case 'kb':
            url = '/kbanswer';
            break;
        case 'answer':
            url = '/answer';
            break;
        default:
            url = '/answer';
    }

    xhr.open('POST', url);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onload = function() {
        if (xhr.status === 200) {
            let response = JSON.parse(xhr.responseText);
            displayMessage('assistant', response.message);
        } else {
            displayMessage('assistant', 'Error: Unable to process your request.');
        }
    };
    xhr.send(JSON.stringify({message: message}));

    // Clear the input field
    messageInput.value = '';
}

function displayMessage(sender, message) {
    let chatContainer = document.getElementById('chat-container');
    let messageDiv = document.createElement('div');

    if (sender === 'assistant') {
        messageDiv.classList.add('assistant-message');
        messageDiv.innerHTML = "<b>ChatWise:</b> " + message;
    } else {
        messageDiv.classList.add('user-message');
        messageDiv.innerHTML = "<b>User:</b> " + message;
    }

    // Create a timestamp element
    let timestamp = document.createElement('span');
    timestamp.classList.add('timestamp');
    let currentTime = new Date().toLocaleTimeString();
    timestamp.innerText = " [" + currentTime + "]";
    messageDiv.appendChild(timestamp);

    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight; // Auto-scroll

    // Implement loading indicators
    setTimeout(() => {
        displayMessage('assistant', '...');
    }, 1000);
}

// Clear chat history
document.getElementById('clear-btn').addEventListener('click', function() {
    document.getElementById('chat-container').innerHTML = '';
});
document.getElementById('send-btn').addEventListener('click', sendMessage);
