(() => {
  const $ = (id) => document.getElementById(id);
  const elements = {
    chatList: $("chatList"), chatTitle: $("chatTitle"), messages: $("messages"),
    welcome: $("welcome"), model: $("modelSelect"), mode: $("modeSelect"),
    target: $("targetSelect"), variations: $("variationSelect"),
    faceIdEnabled: $("faceIdEnabled"), faceIdWeight: $("faceIdWeight"),
    faceIdWeightValue: $("faceIdWeightValue"),
    prompt: $("promptInput"), send: $("sendButton"), imageInput: $("imageInput"),
    attachment: $("attachment"), attachmentPreview: $("attachmentPreview"),
    attachmentName: $("attachmentName"), status: $("connectionStatus"),
    statusDot: document.querySelector(".status-dot"), toast: $("toast"),
    updateStatus: $("updateStatus"), checkUpdates: $("checkUpdates"),
    updateNow: $("updateNow")
  };
  let chats = [];
  let activeId = null;
  let backendUrl = null;
  let assetsUrl = null;
  let saving = Promise.resolve();
  let toastTimer;

  const newId = () => globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const activeChat = () => chats.find((chat) => chat.id === activeId);
  const imageUrl = (url) => url.startsWith("asset:")
    ? `${assetsUrl}/assets/${encodeURIComponent(url.slice("asset:".length))}`
    : `${backendUrl}${url}`;

  function showToast(message) {
    elements.toast.textContent = message;
    elements.toast.classList.remove("hidden");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => elements.toast.classList.add("hidden"), 5000);
  }

  function persist() {
    const snapshot = JSON.stringify(chats);
    saving = saving.then(() => window.pywebview.api.save_chats(snapshot)).catch((error) => {
      showToast(`Could not save chats: ${error.message || error}`);
    });
    return saving;
  }

  function createChat() {
    const chat = {
      id: newId(),
      title: "New chat",
      createdAt: new Date().toISOString(),
      messages: [],
      imageId: null,
      sourceUrl: null,
      sourceName: null,
      modelId: "",
      mode: "in_place_swap",
      target: "dress",
      variations: 1,
      enableFaceId: true,
      faceIdWeight: 0.80
    };
    chats.unshift(chat);
    selectChat(chat.id);
    persist();
  }

  function selectChat(id) {
    activeId = id;
    const chat = activeChat();
    if (!chat) return;
    elements.chatTitle.textContent = chat.title || "New chat";
    elements.model.value = chat.modelId || elements.model.value;
    elements.mode.value = chat.mode || "in_place_swap";
    elements.target.value = chat.target || "dress";
    elements.variations.value = String(chat.variations || 1);
    elements.faceIdEnabled.checked = chat.enableFaceId !== false;
    elements.faceIdWeight.value = String(chat.faceIdWeight || 0.80);
    elements.faceIdWeightValue.textContent = Number(elements.faceIdWeight.value).toFixed(2);
    renderChatList();
    renderMessages();
    renderAttachment();
  }

  function renderChatList() {
    elements.chatList.replaceChildren();
    for (const chat of chats) {
      const button = document.createElement("button");
      button.className = `chat-item${chat.id === activeId ? " active" : ""}`;
      button.textContent = chat.title || "New chat";
      button.title = button.textContent;
      button.addEventListener("click", () => selectChat(chat.id));
      elements.chatList.append(button);
    }
  }

  function appendImage(container, url, label, imageClass = "") {
    const link = document.createElement("a");
    link.href = url;
    link.target = "_blank";
    link.rel = "noreferrer";
    const image = document.createElement("img");
    image.src = url;
    image.alt = label;
    image.loading = "lazy";
    if (imageClass) image.classList.add(imageClass);
    link.append(image);
    container.append(link);
  }

  function renderMessages() {
    const chat = activeChat();
    elements.messages.replaceChildren();
    if (!chat || chat.messages.length === 0) {
      elements.messages.append(elements.welcome);
      elements.welcome.classList.remove("hidden");
      return;
    }
    elements.welcome.classList.add("hidden");
    for (const message of chat.messages) {
      const wrapper = document.createElement("article");
      wrapper.className = "message";
      const meta = document.createElement("div");
      meta.className = "message-meta";
      meta.textContent = message.role === "user" ? "You" : "Face Studio";
      wrapper.append(meta);
      if (message.role === "user") {
        const content = document.createElement("div");
        content.className = "user-message";
        content.textContent = message.text || "";
        wrapper.append(content);
        if (message.sourceUrl) appendImage(content, imageUrl(message.sourceUrl), "Reference photo", "source-thumb");
      } else {
        if (message.error) {
          const error = document.createElement("div");
          error.className = "user-message error-message";
          error.textContent = message.error;
          wrapper.append(error);
        }
        if (Array.isArray(message.results) && message.results.length) {
          const grid = document.createElement("div");
          grid.className = "result-grid";
          message.results.forEach((url, index) => appendImage(grid, imageUrl(url), `Generated result ${index + 1}`));
          wrapper.append(grid);
          const note = document.createElement("div");
          note.className = "assistant-note";
          note.textContent = [message.modelName, message.modeName, message.elapsed ? `${message.elapsed}s` : ""].filter(Boolean).join(" · ");
          wrapper.append(note);
        } else if (message.pending) {
          const pending = document.createElement("div");
          pending.className = "assistant-note";
          pending.textContent = message.pending;
          wrapper.append(pending);
        }
      }
      elements.messages.append(wrapper);
    }
    elements.messages.scrollTop = elements.messages.scrollHeight;
  }

  function renderAttachment() {
    const chat = activeChat();
    if (!chat?.sourceUrl) {
      elements.attachment.classList.add("hidden");
      return;
    }
    elements.attachmentPreview.src = imageUrl(chat.sourceUrl);
    elements.attachmentName.textContent = chat.sourceName || "Ready to use";
    elements.attachment.classList.remove("hidden");
  }

  function saveSettings() {
    const chat = activeChat();
    if (!chat) return;
    chat.modelId = elements.model.value;
    chat.mode = elements.mode.value;
    chat.target = elements.target.value;
    chat.variations = Number(elements.variations.value);
    chat.enableFaceId = elements.faceIdEnabled.checked;
    chat.faceIdWeight = Number(elements.faceIdWeight.value);
    elements.faceIdWeightValue.textContent = chat.faceIdWeight.toFixed(2);
    persist();
  }

  async function uploadImage(file) {
    if (!file || file.size > 50 * 1024 * 1024) {
      showToast("Choose an image smaller than 50 MB.");
      return;
    }
    const chat = activeChat();
    if (!chat) return;
    const data = new FormData();
    data.append("image", file);
    $("attachPhoto").disabled = true;
    elements.status.textContent = "Analyzing reference photo…";
    try {
      const response = await fetch(`${backendUrl}/api/upload`, { method: "POST", body: data });
      const payload = await response.json();
      if (!response.ok || !payload.image_id) throw new Error(payload.error || "Photo upload failed.");
      if (Array.isArray(payload.warnings) && payload.warnings.length) {
        showToast(payload.warnings.join(" "));
      }
      chat.imageId = payload.image_id;
      const sourceUrl = payload.source_url || payload.image_url;
      try {
        chat.sourceUrl = await window.pywebview.api.save_asset(sourceUrl);
      } catch (error) {
        chat.sourceUrl = sourceUrl;
        showToast(`Photo uploaded, but could not preserve it in chat history: ${error.message || error}`);
      }
      chat.sourceName = file.name;
      renderAttachment();
      renderMessages();
      await persist();
    } catch (error) {
      showToast(error.message || "Photo upload failed.");
    } finally {
      $("attachPhoto").disabled = false;
      elements.status.textContent = "Local studio ready";
    }
  }

  async function generate() {
    const chat = activeChat();
    const prompt = elements.prompt.value.trim();
    if (!chat || !prompt) return;
    if (!chat.imageId) {
      showToast("Attach a reference photo before generating.");
      return;
    }
    saveSettings();
    chat.title = chat.title === "New chat" ? prompt.slice(0, 42) : chat.title;
    const userMessage = { role: "user", text: prompt, sourceUrl: chat.sourceUrl, createdAt: new Date().toISOString() };
    const pending = { role: "assistant", pending: "Creating your image…", results: [] };
    chat.messages.push(userMessage, pending);
    elements.prompt.value = "";
    elements.send.disabled = true;
    elements.updateNow.disabled = true;
    renderChatList();
    renderMessages();
    persist();
    const requestId = newId();
    let progressTimer;
    try {
      if (chat.sourceUrl?.startsWith("asset:")) {
        pending.pending = "Restoring reference photo…";
        renderMessages();
        chat.imageId = await window.pywebview.api.restore_source(chat.sourceUrl);
      }
      progressTimer = setInterval(async () => {
        try {
          const progressResponse = await fetch(
            `${backendUrl}/api/outfit-progress?id=${encodeURIComponent(requestId)}`
          );
          if (!progressResponse.ok) return;
          const progress = await progressResponse.json();
          pending.pending = progress.stage || "Creating your image…";
          if (activeId === chat.id) renderMessages();
        } catch {
          // The generation request itself carries errors back to the conversation.
        }
      }, 1500);
      const response = await fetch(`${backendUrl}/api/change-dress`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_id: chat.imageId,
          prompt,
          model_id: chat.modelId || elements.model.value,
          mode: chat.mode,
          target: chat.target,
          variations: chat.variations,
          enable_face_id: chat.enableFaceId,
          face_id_weight: chat.faceIdWeight,
          request_id: requestId
        })
      });
      const result = await response.json();
      if (!response.ok || !result.success) throw new Error(result.error || "Image generation failed.");
      delete pending.pending;
      const resultUrls = (result.results?.length ? result.results : [result.output_url])
        .filter((url) => typeof url === "string" && url);
      if (!resultUrls.length) throw new Error("Image generation completed without returning an image.");
      let savedResults = resultUrls;
      try {
        savedResults = await window.pywebview.api.save_assets(JSON.stringify(resultUrls));
      } catch (error) {
        showToast(`Image generated, but could not preserve it in chat history: ${error.message || error}`);
      }
      Object.assign(pending, {
        results: savedResults,
        modelName: result.model_name,
        modeName: result.mode === "in_place_swap" ? "Outfit swap" : "Full scene",
        elapsed: result.elapsed,
        createdAt: new Date().toISOString()
      });
    } catch (error) {
      delete pending.pending;
      pending.error = error.message || "Image generation failed.";
      pending.createdAt = new Date().toISOString();
    } finally {
      if (progressTimer) clearInterval(progressTimer);
      elements.send.disabled = false;
      elements.updateNow.disabled = false;
      renderChatList();
      renderMessages();
      await persist();
      elements.prompt.focus();
    }
  }

  async function checkForUpdates(silent = false) {
    elements.checkUpdates.disabled = true;
    elements.updateStatus.textContent = "Checking for updates…";
    try {
      const result = await window.pywebview.api.check_for_updates();
      if (!result.success) throw new Error(result.error || "Update check failed.");
      if (result.update_available) {
        elements.updateStatus.textContent = `Version ${result.version} is ready`;
        elements.checkUpdates.classList.add("hidden");
        elements.updateNow.textContent = `Update to ${result.version}`;
        elements.updateNow.classList.remove("hidden");
        elements.updateNow.dataset.version = result.version;
      } else if (!result.release_published) {
        elements.updateStatus.textContent = "No Windows release published";
        elements.updateNow.classList.add("hidden");
        elements.checkUpdates.classList.remove("hidden");
      } else {
        elements.updateStatus.textContent = `Up to date · v${result.version || result.current_version}`;
        elements.updateNow.classList.add("hidden");
        elements.checkUpdates.classList.remove("hidden");
      }
    } catch (error) {
      elements.updateStatus.textContent = "Update check unavailable";
      if (!silent) showToast(error.message || "Could not check for updates.");
    } finally {
      elements.checkUpdates.disabled = false;
    }
  }

  async function initialize() {
    try {
      elements.status.textContent = "Starting local image engine…";
      const started = await window.pywebview.api.start_backend();
      backendUrl = started.url;
      assetsUrl = started.assets_url;
      const response = await fetch(`${backendUrl}/api/status`);
      const status = await response.json();
      elements.model.replaceChildren();
      for (const model of status.models_list || []) {
        const option = document.createElement("option");
        option.value = model.id;
        option.textContent = model.name;
        elements.model.append(option);
      }
      if (!elements.model.options.length) throw new Error("The studio did not report any available image models.");
      chats = await window.pywebview.api.load_chats();
      if (!Array.isArray(chats)) chats = [];
      if (!chats.length) throw new Error("Chat history did not include a conversation.");
      selectChat(chats[0].id);
      elements.status.textContent = status.backend_connected
        ? "Local studio ready"
        : "Studio available · image engine starting";
      elements.statusDot.classList.toggle("ready", Boolean(status.backend_connected));
      checkForUpdates(true);
    } catch (error) {
      elements.status.textContent = "Studio unavailable";
      showToast(error.message || "Could not start Face Studio.");
    }
  }

  $("newChat").addEventListener("click", createChat);
  $("attachPhoto").addEventListener("click", () => $("imageInput").click());
  $("imageInput").addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) uploadImage(file);
    event.target.value = "";
  });
  $("removeAttachment").addEventListener("click", () => {
    const chat = activeChat();
    if (!chat) return;
    chat.imageId = null;
    chat.sourceUrl = null;
    chat.sourceName = null;
    renderAttachment();
    persist();
  });
  $("sendButton").addEventListener("click", generate);
  document.querySelectorAll(".prompt-card").forEach((card) => {
    card.addEventListener("click", () => {
      elements.prompt.value = card.dataset.prompt || "";
      elements.prompt.focus();
      elements.prompt.setSelectionRange(elements.prompt.value.length, elements.prompt.value.length);
    });
  });
  elements.checkUpdates.addEventListener("click", () => checkForUpdates());
  elements.updateNow.addEventListener("click", async () => {
    elements.updateNow.disabled = true;
    elements.updateNow.textContent = "Downloading update…";
    elements.updateStatus.textContent = "Verifying installer";
    try {
      const result = await window.pywebview.api.install_update();
      if (!result.success) throw new Error(result.error || "Could not prepare the update.");
      elements.updateStatus.textContent = `Opening installer · v${result.version}`;
      elements.updateNow.textContent = "Installer opened";
      showToast(`The installer for version ${result.version} is open. Complete setup to apply the update.`);
    } catch (error) {
      elements.updateNow.disabled = false;
      elements.updateNow.textContent = `Update to ${elements.updateNow.dataset.version}`;
      elements.updateStatus.textContent = "Update could not be installed";
      showToast(error.message || "Could not install the update.");
    }
  });
  elements.prompt.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      generate();
    }
  });
  [elements.model, elements.mode, elements.target, elements.variations, elements.faceIdEnabled, elements.faceIdWeight]
    .forEach((control) => control.addEventListener("change", saveSettings));
  elements.faceIdWeight.addEventListener("input", () => {
    elements.faceIdWeightValue.textContent = Number(elements.faceIdWeight.value).toFixed(2);
  });
  $("renameChat").addEventListener("click", async () => {
    const chat = activeChat();
    if (!chat) return;
    const title = window.prompt("Rename this chat", chat.title);
    if (title?.trim()) {
      chat.title = title.trim().slice(0, 80);
      elements.chatTitle.textContent = chat.title;
      renderChatList();
      await persist();
    }
  });
  $("deleteChat").addEventListener("click", async () => {
    const chat = activeChat();
    if (!chat || !window.confirm(`Delete “${chat.title}”? This only removes its chat history.`)) return;
    chats = chats.filter((item) => item.id !== chat.id);
    if (chats.length) selectChat(chats[0].id);
    else createChat();
    await persist();
  });
  window.addEventListener("pywebviewready", initialize, { once: true });
})();
