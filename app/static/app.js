document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const statusDot = document.getElementById('statusDot');
  const statusLabel = document.getElementById('statusLabel');
  const statChunks = document.getElementById('statChunks');
  const statDocs = document.getElementById('statDocs');
  const statModel = document.getElementById('statModel');

  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const uploadProgress = document.getElementById('uploadProgress');

  const pasteSourceName = document.getElementById('pasteSourceName');
  const pasteTextContent = document.getElementById('pasteTextContent');
  const btnPasteSubmit = document.getElementById('btnPasteSubmit');

  const documentList = document.getElementById('documentList');
  const btnClearAll = document.getElementById('btnClearAll');

  const chatMessages = document.getElementById('chatMessages');
  const chatForm = document.getElementById('chatForm');
  const messageInput = document.getElementById('messageInput');
  const btnSend = document.getElementById('btnSend');
  const btnNewChat = document.getElementById('btnNewChat');
  const topKSelect = document.getElementById('topKSelect');

  // State
  let conversationHistory = [];
  let isSending = false;

  // 1. Initial Health & Document Fetch
  async function refreshStats() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        statusDot.className = 'status-indicator online';
        statusLabel.textContent = data.openai_configured ? 'Online (OpenAI)' : 'Online (Demo Mode)';
        statChunks.textContent = data.total_chunks;
        statDocs.textContent = data.sources_count;
        statModel.textContent = data.chat_model || 'gpt-4o-mini';
      } else {
        markOffline();
      }
    } catch (e) {
      markOffline();
    }
  }

  function markOffline() {
    statusDot.className = 'status-indicator';
    statusLabel.textContent = 'Disconnected';
  }

  async function refreshDocuments() {
    try {
      const res = await fetch('/api/documents');
      if (res.ok) {
        const data = await res.json();
        renderDocuments(data.documents || []);
      }
    } catch (e) {
      console.error('Failed to load documents:', e);
    }
  }

  function renderDocuments(docs) {
    if (!docs || docs.length === 0) {
      documentList.innerHTML = '<div class="empty-state">No documents indexed yet.</div>';
      return;
    }

    documentList.innerHTML = '';
    docs.forEach(doc => {
      const item = document.createElement('div');
      item.className = 'doc-item';
      item.innerHTML = `
        <div class="doc-meta">
          <span class="doc-name" title="${doc.source}">${doc.source}</span>
          <span class="doc-chunks">${doc.chunks_count} chunk${doc.chunks_count === 1 ? '' : 's'}</span>
        </div>
        <button class="btn-doc-delete" data-source="${encodeURIComponent(doc.source)}" title="Delete document">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="3 6 5 6 21 6"/>
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
          </svg>
        </button>
      `;

      item.querySelector('.btn-doc-delete').addEventListener('click', async (e) => {
        const src = e.currentTarget.getAttribute('data-source');
        if (confirm(`Remove "${decodeURIComponent(src)}" from the knowledge base?`)) {
          await deleteDocument(src);
        }
      });

      documentList.appendChild(item);
    });
  }

  async function deleteDocument(encodedSource) {
    try {
      const res = await fetch(`/api/documents/${encodedSource}`, { method: 'DELETE' });
      if (res.ok) {
        await refreshStats();
        await refreshDocuments();
      } else {
        alert('Failed to delete document.');
      }
    } catch (e) {
      alert('Network error while deleting document.');
    }
  }

  // 2. Clear All Knowledge
  btnClearAll.addEventListener('click', async () => {
    if (confirm('Are you sure you want to purge all documents and vector embeddings? This cannot be undone.')) {
      try {
        const res = await fetch('/api/clear', { method: 'POST' });
        if (res.ok) {
          await refreshStats();
          await refreshDocuments();
        }
      } catch (e) {
        alert('Failed to clear knowledge base.');
      }
    }
  });

  // 3. Dropzone Upload
  dropzone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      uploadFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files.length > 0) {
      uploadFile(fileInput.files[0]);
    }
  });

  async function uploadFile(file) {
    uploadProgress.textContent = `Uploading ${file.name}...`;
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/ingest/file', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (res.ok) {
        uploadProgress.textContent = `✓ Indexed ${data.chunks_added} chunks!`;
        setTimeout(() => { uploadProgress.textContent = ''; }, 3500);
        await refreshStats();
        await refreshDocuments();
      } else {
        uploadProgress.textContent = `Error: ${data.detail || 'Upload failed'}`;
      }
    } catch (e) {
      uploadProgress.textContent = 'Upload failed. Check server status.';
    } finally {
      fileInput.value = '';
    }
  }

  // 4. Paste Text Snippet
  btnPasteSubmit.addEventListener('click', async () => {
    const text = pasteTextContent.value.trim();
    const source = pasteSourceName.value.trim() || 'manual_snippet.txt';
    if (!text) {
      alert('Please enter some text to index.');
      return;
    }

    btnPasteSubmit.disabled = true;
    btnPasteSubmit.textContent = 'Indexing...';

    try {
      const res = await fetch('/api/ingest/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, source_name: source })
      });
      const data = await res.json();
      if (res.ok) {
        pasteTextContent.value = '';
        pasteSourceName.value = '';
        await refreshStats();
        await refreshDocuments();
        alert(`Successfully indexed ${data.chunks_added} chunks from "${source}"!`);
      } else {
        alert(`Error: ${data.detail || 'Ingestion failed'}`);
      }
    } catch (e) {
      alert('Network error while indexing text.');
    } finally {
      btnPasteSubmit.disabled = false;
      btnPasteSubmit.textContent = 'Index Text Snippet';
    }
  });

  // 5. Chat Interaction
  function appendMessage(role, content, citations = []) {
    const row = document.createElement('div');
    row.className = `message-row ${role}`;

    const avatar = document.createElement('div');
    avatar.className = `avatar ${role}-avatar`;
    avatar.innerHTML = role === 'assistant' 
      ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
           <circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>
         </svg>`
      : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
           <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>
         </svg>`;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    if (role === 'assistant' && window.marked) {
      contentDiv.innerHTML = marked.parse(content);
    } else {
      contentDiv.textContent = content;
    }
    bubble.appendChild(contentDiv);

    // Citations drawer
    if (citations && citations.length > 0) {
      const citationsBox = document.createElement('div');
      citationsBox.className = 'citations-box';
      citationsBox.innerHTML = `
        <div class="citations-header">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
          Sources & Citations (${citations.length})
        </div>
        <div class="citation-cards">
          ${citations.map(c => `
            <div class="citation-card">
              <div class="citation-card-header">
                <span class="citation-source" title="${c.source}">[${c.citation_id}] ${c.source}</span>
                <span class="citation-badge">${Math.round(c.score * 100)}% match</span>
              </div>
              <div class="citation-snippet" title="${c.snippet}">"${c.snippet}"</div>
            </div>
          `).join('')}
        </div>
      `;
      bubble.appendChild(citationsBox);
    }

    row.appendChild(avatar);
    row.appendChild(bubble);
    chatMessages.appendChild(row);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return row;
  }

  function appendTypingIndicator() {
    const row = document.createElement('div');
    row.className = 'message-row assistant typing-row';
    row.innerHTML = `
      <div class="avatar assistant-avatar">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>
        </svg>
      </div>
      <div class="message-bubble">
        <div class="typing-dots">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
    `;
    chatMessages.appendChild(row);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return row;
  }

  async function handleSendMessage(query) {
    if (!query || isSending) return;
    isSending = true;
    btnSend.disabled = true;

    // 1. Add user bubble
    appendMessage('user', query);
    conversationHistory.push({ role: 'user', content: query });
    messageInput.value = '';
    adjustTextareaHeight();

    // 2. Add loading animation
    const typingIndicator = appendTypingIndicator();

    try {
      const topK = parseInt(topKSelect.value, 10) || 4;
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: query,
          history: conversationHistory.slice(-6),
          top_k: topK
        })
      });

      typingIndicator.remove();

      if (res.ok) {
        const data = await res.json();
        appendMessage('assistant', data.answer, data.citations);
        conversationHistory.push({ role: 'assistant', content: data.answer });
      } else {
        const err = await res.json();
        appendMessage('assistant', `⚠️ **Error**: ${err.detail || 'Failed to generate response.'}`);
      }
    } catch (e) {
      typingIndicator.remove();
      appendMessage('assistant', '⚠️ **Connection Error**: Unable to contact the FastAPI server.');
    } finally {
      isSending = false;
      btnSend.disabled = false;
      messageInput.focus();
    }
  }

  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    handleSendMessage(messageInput.value.trim());
  });

  // Shift+Enter for newline, Enter to send
  messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(messageInput.value.trim());
    }
  });

  function adjustTextareaHeight() {
    messageInput.style.height = 'auto';
    messageInput.style.height = `${Math.min(messageInput.scrollHeight, 140)}px`;
  }
  messageInput.addEventListener('input', adjustTextareaHeight);

  // Preset chips click
  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const prompt = chip.getAttribute('data-prompt');
      if (prompt) {
        handleSendMessage(prompt);
      }
    });
  });

  // Reset chat
  btnNewChat.addEventListener('click', () => {
    conversationHistory = [];
    chatMessages.innerHTML = `
      <div class="message-row assistant">
        <div class="avatar assistant-avatar">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>
          </svg>
        </div>
        <div class="message-bubble">
          <div class="message-content">
            <p>Chat reset! Ask any question or upload new documents on the left.</p>
          </div>
        </div>
      </div>
    `;
  });

  // Initial load
  refreshStats();
  refreshDocuments();
});
