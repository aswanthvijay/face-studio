/**
 * Face Studio AI — Web Client Application Logic
 * Supports Outfit & Wardrobe Studio, AI Restoration, Rembg, Magic Object Eraser, 4K Upscaler, and Deblur.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements - Header & Status
  const statusPill = document.getElementById('statusPill');
  const statusText = document.getElementById('statusText');
  const downloadBtn = document.getElementById('downloadBtn');
  const canvasTag = document.getElementById('canvasTag');
  const canvasPlaceholder = document.getElementById('canvasPlaceholder');
  const canvasLoading = document.getElementById('canvasLoading');
  const loadingStatusText = document.getElementById('loadingStatusText');
  const loadingTimer = document.getElementById('loadingTimer');
  const splitSliderWrapper = document.getElementById('splitSliderWrapper');
  const beforeLayer = document.getElementById('beforeLayer');
  const beforeImg = document.getElementById('beforeImg');
  const afterImg = document.getElementById('afterImg');
  const sliderDivider = document.getElementById('sliderDivider');
  const viewerActions = document.getElementById('viewerActions');

  const generationInfo = document.getElementById('generationInfo');
  const infoModel = document.getElementById('infoModel');
  const infoMode = document.getElementById('infoMode');
  const infoTime = document.getElementById('infoTime');
  const infoLikeness = document.getElementById('infoLikeness');

  // Navigation Tabs
  const navTabs = document.querySelectorAll('.nav-tab');
  const tabPanels = document.querySelectorAll('.tab-panel');

  // Outfit Studio Controls
  const modeInPlace = document.getElementById('modeInPlace');
  const modeFullScene = document.getElementById('modeFullScene');
  const targetGarmentSection = document.getElementById('targetGarmentSection');
  const framingSection = document.getElementById('framingSection');
  const fileInput = document.getElementById('fileInput');
  const dropzone = document.getElementById('dropzone');
  const dropzoneContent = document.getElementById('dropzoneContent');
  const uploadPreviewContainer = document.getElementById('uploadPreviewContainer');
  const uploadPreviewImg = document.getElementById('uploadPreviewImg');
  const removeUploadBtn = document.getElementById('removeUploadBtn');
  const loadSampleBtn = document.getElementById('loadSampleBtn');
  const modelSelect = document.getElementById('modelSelect');
  const modelDesc = document.getElementById('modelDesc');
  const loraSelect = document.getElementById('loraSelect');
  const loraControls = document.getElementById('loraControls');
  const loraWeightSlider = document.getElementById('loraWeightSlider');
  const loraWeightVal = document.getElementById('loraWeightVal');
  const triggerTagRow = document.getElementById('triggerTagRow');
  const triggerTagBtn = document.getElementById('triggerTagBtn');
  const loraDesc = document.getElementById('loraDesc');
  const promptInput = document.getElementById('promptInput');
  const generateBtn = document.getElementById('generateBtn');
  const enableFaceIdCheckbox = document.getElementById('enableFaceIdCheckbox');
  const faceIdWeightSlider = document.getElementById('faceIdWeightSlider');
  const faceIdWeightVal = document.getElementById('faceIdWeightVal');
  const previewOverlayTag = document.getElementById('previewOverlayTag');
  const sloganSection = document.getElementById('sloganSection');
  const sloganInput = document.getElementById('sloganInput');
  const sloganHint = document.getElementById('sloganHint');

  // Tool Action Buttons & Inputs
  const restoreBtn = document.getElementById('restoreBtn');
  const restoreFidelitySlider = document.getElementById('restoreFidelitySlider');
  const restoreFidelityVal = document.getElementById('restoreFidelityVal');

  const rembgBtn = document.getElementById('rembgBtn');

  const eraseBtn = document.getElementById('eraseBtn');
  const clearBrushBtn = document.getElementById('clearBrushBtn');
  const brushSizeSlider = document.getElementById('brushSizeSlider');
  const brushSizeVal = document.getElementById('brushSizeVal');
  const eraserBaseCanvas = document.getElementById('eraserBaseCanvas');
  const eraserDrawCanvas = document.getElementById('eraserDrawCanvas');

  const upscaleBtn = document.getElementById('upscaleBtn');
  const upscaleModelSelect = document.getElementById('upscaleModelSelect');

  const deblurBtn = document.getElementById('deblurBtn');
  const deblurStrengthSlider = document.getElementById('deblurStrengthSlider');
  const deblurStrengthVal = document.getElementById('deblurStrengthVal');

  // Global State
  let currentActiveTab = 'panel-outfit';
  let currentMode = 'in_place_swap';
  let currentTarget = 'dress';
  let currentFraming = 'editorial';
  let currentAspect = 'portrait';
  let currentImageId = null;
  let currentSourceUrl = null;
  let currentOutputUrl = null;
  let isDraggingSlider = false;
  let timerInterval = null;
  let loraRegistry = {};

  // Eraser Canvas Drawing State
  let isDrawingOnCanvas = false;
  let eraserBrushRadius = 24;
  let eraserBaseImgLoaded = null;

  // Model Descriptions (Streamlined Top 3)
  const MODEL_DESCS = {
    cyberrealistic_xl: "True DSLR micro-pores, natural epidermal depth, zero AI plastic smoothing.",
    realvis_xl: "Razor-sharp photorealism, lifelike eye catchlights, and cinematic studio lighting.",
    flux_klein: "Next-gen Klein DiT with Qwen3 Heretic text encoder for unrestricted wardrobe and scene styling."
  };

  // -------------------------------------------------------------------------
  // 1. Tab Switching Logic
  // -------------------------------------------------------------------------
  navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      navTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const targetPanelId = tab.dataset.tab;
      currentActiveTab = targetPanelId;

      tabPanels.forEach(panel => {
        if (panel.id === targetPanelId) {
          panel.classList.remove('hidden');
        } else {
          panel.classList.add('hidden');
        }
      });

      // If switched to eraser tab and image is available, initialize canvas after DOM layout
      if (targetPanelId === 'panel-eraser' && currentSourceUrl) {
        requestAnimationFrame(() => {
          initEraserCanvas(currentSourceUrl);
        });
      }
    });
  });

  // -------------------------------------------------------------------------
  // 2. Synchronize Uploaded Image across all Tool Panels
  // -------------------------------------------------------------------------
  function setGlobalImage(imageId, url) {
    currentImageId = imageId;
    currentSourceUrl = url;

    // Update Outfit Studio dropzone
    if (uploadPreviewImg && uploadPreviewContainer && dropzoneContent) {
      uploadPreviewImg.src = url;
      uploadPreviewContainer.classList.remove('hidden');
      dropzoneContent.classList.add('hidden');
    }

    // Update all tool dropzones
    document.querySelectorAll('.tool-dropzone').forEach(dz => {
      const pImg = dz.querySelector('.tool-preview-img');
      const pContainer = dz.querySelector('.upload-preview-container');
      const content = dz.querySelector('.dropzone-content');
      if (pImg) pImg.src = url;
      if (pContainer) pContainer.classList.remove('hidden');
      if (content) content.classList.add('hidden');
    });

    // If on eraser panel, setup canvas
    if (currentActiveTab === 'panel-eraser') {
      initEraserCanvas(url);
    }
  }

  function clearGlobalImage() {
    currentImageId = null;
    currentSourceUrl = null;

    if (uploadPreviewContainer && dropzoneContent && fileInput) {
      uploadPreviewContainer.classList.add('hidden');
      dropzoneContent.classList.remove('hidden');
      fileInput.value = '';
    }

    document.querySelectorAll('.tool-dropzone').forEach(dz => {
      const pContainer = dz.querySelector('.upload-preview-container');
      const content = dz.querySelector('.dropzone-content');
      const fInput = dz.querySelector('.tool-file-input');
      if (pContainer) pContainer.classList.add('hidden');
      if (content) content.classList.remove('hidden');
      if (fInput) fInput.value = '';
    });
  }

  // -------------------------------------------------------------------------
  // 3. Upload Event Handlers (Outfit dropzone & Tool dropzones)
  // -------------------------------------------------------------------------
  async function handleImageUpload(file) {
    if (!file || file.size > 50 * 1024 * 1024) {
      alert("Choose an image smaller than 50 MB.");
      return;
    }
    const formData = new FormData();
    formData.append('image', file);
    try {
      if (statusText) statusText.textContent = "Analyzing & Segmenting Garment...";
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (data.image_id) {
        setGlobalImage(data.image_id, data.source_url || data.image_url);
        if (previewOverlayTag && data.coverage !== undefined) {
          const faceCount = Number(data.faces_detected || 0);
          const faceNote = faceCount > 1 ? ` · largest of ${faceCount} people` : "";
          previewOverlayTag.textContent = `Garment Mask: ${(data.coverage * 100).toFixed(0)}%${faceNote}`;
        }
      } else {
        alert("Upload error: " + (data.error || "Failed to process image"));
      }
    } catch (e) {
      alert("Network error during upload: " + e.message);
    } finally {
      if (statusText) statusText.textContent = "GPU Engine Active";
    }
  }

  // Outfit Studio dropzone events
  if (dropzone && fileInput) {
    dropzone.addEventListener('click', (e) => {
      if (e.target !== removeUploadBtn && !e.target.closest('#removeUploadBtn') && !e.target.closest('#loadSampleBtn')) {
        fileInput.click();
      }
    });
    fileInput.addEventListener('change', () => {
      if (fileInput.files.length) handleImageUpload(fileInput.files[0]);
    });
    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('drag-over'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag-over');
      if (e.dataTransfer.files.length) handleImageUpload(e.dataTransfer.files[0]);
    });
  }

  if (removeUploadBtn) {
    removeUploadBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      clearGlobalImage();
    });
  }

  // Tool dropzones events
  document.querySelectorAll('.tool-dropzone').forEach(dz => {
    const fInput = dz.querySelector('.tool-file-input');
    const rmBtn = dz.querySelector('.tool-remove-btn');
    const sampleBtn = dz.querySelector('.tool-sample-btn');

    dz.addEventListener('click', (e) => {
      if (e.target !== rmBtn && !e.target.closest('.tool-remove-btn') && !e.target.closest('.tool-sample-btn')) {
        if (fInput) fInput.click();
      }
    });

    if (fInput) {
      fInput.addEventListener('change', () => {
        if (fInput.files.length) handleImageUpload(fInput.files[0]);
      });
    }

    if (rmBtn) {
      rmBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearGlobalImage();
      });
    }

    if (sampleBtn) {
      sampleBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        await loadStudioSample();
      });
    }
  });

  async function loadStudioSample() {
    try {
      const res = await fetch('/api/sample');
      const data = await res.json();
      if (data.image_id) {
        setGlobalImage(data.image_id, data.image_url);
      }
    } catch (e) {
      console.warn("Sample load error:", e);
    }
  }

  if (loadSampleBtn) {
    loadSampleBtn.addEventListener('click', async (e) => {
      e.stopPropagation();
      await loadStudioSample();
    });
  }

  // -------------------------------------------------------------------------
  // 4. Initial Health & Models Status
  // -------------------------------------------------------------------------
  async function fetchServerStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      if (data.ready) {
        statusPill.className = 'status-pill ready';
        statusText.textContent = 'GPU Engine Active';
      }
      if (data.models_list && modelSelect) {
        const prevVal = modelSelect.value;
        modelSelect.innerHTML = '';
        data.models_list.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m.id;
          opt.textContent = `${m.name} [${m.badge || ''}]`;
          if (m.id === (prevVal || data.default_model)) opt.selected = true;
          modelSelect.appendChild(opt);
        });
        if (modelDesc && MODEL_DESCS[modelSelect.value]) {
          modelDesc.textContent = MODEL_DESCS[modelSelect.value];
        }
      }
      if (data.loras) {
        loraRegistry = data.loras;
        populateLoras(data.loras);
      }
    } catch (e) {
      statusPill.className = 'status-pill error';
      statusText.textContent = 'Engine Offline';
    }
  }
  fetchServerStatus();

  function populateLoras(loras) {
    if (!loraSelect) return;
    loraSelect.innerHTML = '<option value="" selected>None (Base Model Only)</option>';
    Object.values(loras).forEach(lora => {
      const opt = document.createElement('option');
      opt.value = lora.id;
      opt.textContent = `${lora.name} [${lora.badge || 'LoRA'}]`;
      loraSelect.appendChild(opt);
    });
  }

  // -------------------------------------------------------------------------
  // 5. Outfit Controls Listeners
  // -------------------------------------------------------------------------
  if (modeInPlace && modeFullScene) {
    modeInPlace.addEventListener('click', () => {
      currentMode = 'in_place_swap';
      modeInPlace.classList.add('active');
      modeFullScene.classList.remove('active');
      targetGarmentSection.classList.remove('hidden');
      framingSection.classList.add('hidden');
      if (modelSelect && !['cyberrealistic_xl', 'realvis_xl', 'flux_klein'].includes(modelSelect.value)) {
        modelSelect.value = "cyberrealistic_xl";
        if (modelDesc) modelDesc.textContent = MODEL_DESCS["cyberrealistic_xl"];
      }
    });

    modeFullScene.addEventListener('click', () => {
      currentMode = 'full_scene';
      modeFullScene.classList.add('active');
      modeInPlace.classList.remove('active');
      targetGarmentSection.classList.add('hidden');
      framingSection.classList.remove('hidden');
    });
  }

  function updateTypographyGuidance() {
    if (!sloganInput || !sloganHint) return;
    const txt = sloganInput.value.trim();
    const currentModel = modelSelect ? modelSelect.value : 'cyberrealistic_xl';
    const isSdxl = ['cyberrealistic_xl', 'realvis_xl'].includes(currentModel);

    if (txt.length > 0 && isSdxl) {
      sloganHint.innerHTML = "⚠️ <strong style='color: #fbbf24;'>Notice:</strong> SDXL models cannot spell text words. Switch to <strong style='color: #60a5fa;'>Flux-Klein</strong> for typography rendering.";
    } else if (currentModel === 'flux_klein') {
      sloganHint.innerHTML = "✨ Flux-Klein DiT supports accurate typography rendering via Qwen3 text encoder.";
    } else {
      sloganHint.innerHTML = "Flux-Klein renders exact character typography onto garments with 100% correct spelling.";
    }
  }

  if (sloganInput) {
    sloganInput.addEventListener('input', updateTypographyGuidance);
  }

  if (modelSelect) {
    modelSelect.addEventListener('change', () => {
      const m = modelSelect.value;
      if (modelDesc && MODEL_DESCS[m]) modelDesc.textContent = MODEL_DESCS[m];
      updateTypographyGuidance();
    });
  }

  // Sliders
  if (restoreFidelitySlider && restoreFidelityVal) {
    restoreFidelitySlider.addEventListener('input', () => {
      const val = parseFloat(restoreFidelitySlider.value);
      restoreFidelityVal.textContent = val === 0.85 ? "0.85 (Natural Pores)" : val.toFixed(2);
    });
  }

  if (deblurStrengthSlider && deblurStrengthVal) {
    deblurStrengthSlider.addEventListener('input', () => {
      const val = parseFloat(deblurStrengthSlider.value);
      deblurStrengthVal.textContent = val === 1.35 ? "1.35 (Optimal)" : val.toFixed(2);
    });
  }

  if (brushSizeSlider && brushSizeVal) {
    brushSizeSlider.addEventListener('input', () => {
      eraserBrushRadius = parseInt(brushSizeSlider.value, 10);
      brushSizeVal.textContent = `${eraserBrushRadius}px`;
    });
  }

  if (faceIdWeightSlider && faceIdWeightVal) {
    faceIdWeightSlider.addEventListener('input', () => {
      const val = parseFloat(faceIdWeightSlider.value);
      faceIdWeightVal.textContent = val === 0.65 ? "0.65 (Optimal)" : val.toFixed(2);
    });
  }

  if (loraWeightSlider && loraWeightVal) {
    loraWeightSlider.addEventListener('input', () => {
      loraWeightVal.textContent = parseFloat(loraWeightSlider.value).toFixed(2);
    });
  }

  if (loraSelect) {
    loraSelect.addEventListener('change', () => {
      const sel = loraSelect.value;
      if (!sel) {
        loraControls.classList.add('hidden');
        loraDesc.textContent = "Enhance garments with trained bespoke fashion lines, embroidery, or couture fabrics.";
      } else {
        const item = loraRegistry[sel];
        loraControls.classList.remove('hidden');
        if (item) {
          loraDesc.textContent = item.desc || "Custom LoRA adapter.";
          if (item.trigger) {
            triggerTagRow.classList.remove('hidden');
            triggerTagBtn.textContent = `+ Add "${item.trigger}"`;
            triggerTagBtn.onclick = () => {
              if (!promptInput.value.includes(item.trigger)) {
                promptInput.value = promptInput.value.trim() ? `${promptInput.value}, ${item.trigger}` : item.trigger;
              }
            };
          } else {
            triggerTagRow.classList.add('hidden');
          }
        }
      }
    });
  }

  // Chips
  document.querySelectorAll('#garmentChips .target-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('#garmentChips .target-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      currentTarget = chip.dataset.target;
    });
  });

  document.querySelectorAll('#framingChips .target-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('#framingChips .target-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      currentFraming = chip.dataset.framing;
      currentAspect = chip.dataset.aspect || 'portrait';
    });
  });

  document.querySelectorAll('.p-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      promptInput.value = chip.dataset.text;
    });
  });

  // -------------------------------------------------------------------------
  // 6. Magic Object Eraser Canvas Implementation
  // -------------------------------------------------------------------------
  function initEraserCanvas(imgUrl) {
    if (!eraserBaseCanvas || !eraserDrawCanvas) return;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      eraserBaseImgLoaded = img;
      const wrapper = document.getElementById('canvasPaintWrapper');
      const containerW = wrapper ? (wrapper.clientWidth || wrapper.parentElement?.clientWidth || 360) : 360;
      const maxW = Math.max(120, containerW - 20);
      const maxH = 260;
      const scale = Math.min(maxW / img.width, maxH / img.height);
      const w = Math.max(64, Math.round(img.width * scale));
      const h = Math.max(64, Math.round(img.height * scale));

      eraserBaseCanvas.width = w;
      eraserBaseCanvas.height = h;
      eraserDrawCanvas.width = w;
      eraserDrawCanvas.height = h;

      const baseCtx = eraserBaseCanvas.getContext('2d');
      baseCtx.drawImage(img, 0, 0, w, h);

      const drawCtx = eraserDrawCanvas.getContext('2d');
      drawCtx.clearRect(0, 0, w, h);
    };
    img.src = imgUrl;
  }

  if (eraserDrawCanvas) {
    const drawCtx = eraserDrawCanvas.getContext('2d');

    function startDraw(x, y) {
      isDrawingOnCanvas = true;
      drawCtx.beginPath();
      drawCtx.moveTo(x, y);
      drawCtx.lineCap = 'round';
      drawCtx.lineJoin = 'round';
      drawCtx.strokeStyle = 'rgba(239, 68, 68, 0.65)';
      drawCtx.lineWidth = eraserBrushRadius;
    }

    function doDraw(x, y) {
      if (!isDrawingOnCanvas) return;
      drawCtx.lineTo(x, y);
      drawCtx.stroke();
    }

    function endDraw() {
      if (!isDrawingOnCanvas) return;
      isDrawingOnCanvas = false;
      drawCtx.closePath();
    }

    eraserDrawCanvas.addEventListener('mousedown', (e) => {
      const rect = eraserDrawCanvas.getBoundingClientRect();
      startDraw(e.clientX - rect.left, e.clientY - rect.top);
    });
    eraserDrawCanvas.addEventListener('mousemove', (e) => {
      const rect = eraserDrawCanvas.getBoundingClientRect();
      doDraw(e.clientX - rect.left, e.clientY - rect.top);
    });
    eraserDrawCanvas.addEventListener('mouseup', endDraw);
    eraserDrawCanvas.addEventListener('mouseleave', endDraw);

    // Touch support for mobile/tablets
    eraserDrawCanvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      if (!e.touches.length) return;
      const rect = eraserDrawCanvas.getBoundingClientRect();
      startDraw(e.touches[0].clientX - rect.left, e.touches[0].clientY - rect.top);
    });
    eraserDrawCanvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      if (!e.touches.length) return;
      const rect = eraserDrawCanvas.getBoundingClientRect();
      doDraw(e.touches[0].clientX - rect.left, e.touches[0].clientY - rect.top);
    });
    eraserDrawCanvas.addEventListener('touchend', endDraw);
  }

  if (clearBrushBtn && eraserDrawCanvas) {
    clearBrushBtn.addEventListener('click', () => {
      const ctx = eraserDrawCanvas.getContext('2d');
      ctx.clearRect(0, 0, eraserDrawCanvas.width, eraserDrawCanvas.height);
    });
  }

  function getEraserMaskBase64() {
    if (!eraserDrawCanvas || !eraserBaseImgLoaded) return null;
    // Export offscreen canvas at original image resolution
    const origW = eraserBaseImgLoaded.naturalWidth || eraserBaseImgLoaded.width;
    const origH = eraserBaseImgLoaded.naturalHeight || eraserBaseImgLoaded.height;

    const offCanvas = document.createElement('canvas');
    offCanvas.width = origW;
    offCanvas.height = origH;
    const offCtx = offCanvas.getContext('2d');

    // Fill background with black
    offCtx.fillStyle = '#000000';
    offCtx.fillRect(0, 0, origW, origH);

    // Draw user mask strokes scaled up
    offCtx.drawImage(eraserDrawCanvas, 0, 0, origW, origH);

    // Convert colored stroke to pure white mask (threshold)
    const imgData = offCtx.getImageData(0, 0, origW, origH);
    const d = imgData.data;
    let whitePixelCount = 0;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i] > 20 || d[i + 1] > 20 || d[i + 2] > 20) {
        d[i] = 255;
        d[i + 1] = 255;
        d[i + 2] = 255;
        d[i + 3] = 255;
        whitePixelCount++;
      } else {
        d[i] = 0;
        d[i + 1] = 0;
        d[i + 2] = 0;
        d[i + 3] = 255;
      }
    }
    if (whitePixelCount === 0) {
      return null;
    }
    offCtx.putImageData(imgData, 0, 0);
    return offCanvas.toDataURL('image/png');
  }

  // -------------------------------------------------------------------------
  // 7. Universal Result Display & Before/After Slider
  // -------------------------------------------------------------------------
  function startProgress(statusMsg) {
    canvasPlaceholder.classList.add('hidden');
    splitSliderWrapper.classList.add('hidden');
    canvasLoading.classList.remove('hidden');
    viewerActions.classList.add('hidden');
    generationInfo.classList.add('hidden');
    loadingStatusText.textContent = statusMsg;
    loadingTimer.textContent = 'Elapsed: 0s';

    let sec = 0;
    if (timerInterval) clearInterval(timerInterval);
    timerInterval = setInterval(() => {
      sec++;
      loadingTimer.textContent = `Elapsed: ${sec}s`;
    }, 1000);
  }

  function endProgress() {
    if (timerInterval) clearInterval(timerInterval);
    canvasLoading.classList.add('hidden');
  }

  function displayResult(beforeUrl, afterUrl, meta = {}) {
    endProgress();
    currentOutputUrl = afterUrl;

    beforeImg.src = beforeUrl;
    afterImg.src = afterUrl;

    const reveal = () => {
      canvasPlaceholder.classList.add('hidden');
      canvasLoading.classList.add('hidden');
      splitSliderWrapper.classList.remove('hidden');
      viewerActions.classList.remove('hidden');
      generationInfo.classList.remove('hidden');

      if (infoModel) infoModel.textContent = meta.toolName || 'AI Transformation';
      if (infoMode) infoMode.textContent = meta.modeName || 'Default';
      if (infoTime) infoTime.textContent = meta.elapsed ? `${meta.elapsed}s` : 'Instant';
      if (infoLikeness) infoLikeness.textContent = meta.likeness || '100% Locked';
      if (canvasTag) canvasTag.textContent = meta.tag || 'Creation Complete';

      // Special background handling for transparent PNG
      if (meta.isTransparent) {
        splitSliderWrapper.classList.add('checkerboard-bg');
      } else {
        splitSliderWrapper.classList.remove('checkerboard-bg');
      }

      updateSliderPosition(50);
    };

    afterImg.onload = reveal;
    if (afterImg.complete && afterImg.naturalWidth > 0) reveal();
  }

  // -------------------------------------------------------------------------
  // 8. API Execution Handlers
  // -------------------------------------------------------------------------

  // 1. Outfit Studio
  generateBtn.addEventListener('click', async () => {
    if (!currentImageId && !currentSourceUrl) {
      alert("Please upload a portrait reference photo or click 'Use Studio Sample'.");
      return;
    }
    generateBtn.disabled = true;
    const selectedModel = modelSelect ? modelSelect.value : 'flux_klein';
    const engineLabel = selectedModel === 'moss_image' ? 'MOSS-Image 5B DiT' : 'Klein DiT';
    startProgress(`Synthesizing Photorealistic Outfit with ${engineLabel}...`);

    try {
      const payload = {
        image_id: currentImageId,
        mode: currentMode,
        target: currentTarget,
        model_id: selectedModel,
        prompt: promptInput ? promptInput.value : '',
        slogan: sloganInput ? sloganInput.value.trim() : '',
        framing: currentFraming,
        aspect_ratio: currentAspect,
        lora_id: loraSelect ? loraSelect.value : null,
        lora_weight: loraWeightSlider ? parseFloat(loraWeightSlider.value) : 0.75,
        enable_face_id: enableFaceIdCheckbox ? enableFaceIdCheckbox.checked : true,
        face_id_weight: faceIdWeightSlider ? parseFloat(faceIdWeightSlider.value) : 0.65
      };

      const res = await fetch('/api/change-dress', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.success && data.output_url) {
        displayResult(data.source_url, data.output_url, {
          toolName: data.model_name || 'Outfit Studio',
          modeName: currentMode === 'in_place_swap' ? 'In-Place Swap' : 'Full Scene',
          elapsed: data.elapsed,
          likeness: '100% Biometric Lock',
          tag: 'Outfit Complete'
        });
      } else {
        endProgress();
        canvasPlaceholder.classList.remove('hidden');
        alert("Generation failed: " + (data.error || "Unknown server error"));
      }
    } catch (err) {
      endProgress();
      canvasPlaceholder.classList.remove('hidden');
      alert("Network error: " + err.message);
    } finally {
      generateBtn.disabled = false;
    }
  });

  // 2. Photo Restoration
  if (restoreBtn) {
    restoreBtn.addEventListener('click', async () => {
      if (!currentImageId && !currentSourceUrl) {
        alert("Please upload a photo to restore.");
        return;
      }
      restoreBtn.disabled = true;
      startProgress("Restoring Photo & Facial Dermal Micro-Pores...");

      try {
        const payload = {
          image_id: currentImageId,
          fidelity: restoreFidelitySlider ? parseFloat(restoreFidelitySlider.value) : 0.85
        };
        const res = await fetch('/api/restore', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success && data.output_url) {
          displayResult(data.source_url, data.output_url, {
            toolName: 'CodeFormer Prior',
            modeName: `Fidelity ${payload.fidelity}`,
            elapsed: data.elapsed,
            likeness: 'Authentic Dermal Grain',
            tag: 'Restoration Complete'
          });
        } else {
          endProgress();
          canvasPlaceholder.classList.remove('hidden');
          alert("Restoration error: " + (data.error || "Failed to restore photo"));
        }
      } catch (e) {
        endProgress();
        canvasPlaceholder.classList.remove('hidden');
        alert("Network error: " + e.message);
      } finally {
        restoreBtn.disabled = false;
      }
    });
  }

  // 3. Background Remover
  if (rembgBtn) {
    rembgBtn.addEventListener('click', async () => {
      if (!currentImageId && !currentSourceUrl) {
        alert("Please upload a photo for background removal.");
        return;
      }
      rembgBtn.disabled = true;
      startProgress("Extracting Alpha Cutout & Boundary Matting...");

      try {
        const payload = { image_id: currentImageId };
        const res = await fetch('/api/remove-bg', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success && data.output_url) {
          displayResult(data.source_url, data.output_url, {
            toolName: 'rembg Bria-2.0',
            modeName: 'Alpha Transparent Cutout',
            elapsed: data.elapsed,
            likeness: 'Original Subject Preserved',
            tag: 'Background Removed',
            isTransparent: true
          });
        } else {
          endProgress();
          canvasPlaceholder.classList.remove('hidden');
          alert("Background removal error: " + (data.error || "Failed to remove background"));
        }
      } catch (e) {
        endProgress();
        canvasPlaceholder.classList.remove('hidden');
        alert("Network error: " + e.message);
      } finally {
        rembgBtn.disabled = false;
      }
    });
  }

  // 4. Magic Object Eraser
  if (eraseBtn) {
    eraseBtn.addEventListener('click', async () => {
      if (!currentImageId && !currentSourceUrl) {
        alert("Please upload a photo first.");
        return;
      }
      const maskBase64 = getEraserMaskBase64();
      if (!maskBase64) {
        alert("Please paint over the object or blemish you wish to erase.");
        return;
      }
      eraseBtn.disabled = true;
      startProgress("Seamlessly Inpainting & Synthesizing Background Texture...");

      try {
        const payload = {
          image_id: currentImageId,
          mask: maskBase64
        };
        const res = await fetch('/api/erase-object', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success && data.output_url) {
          displayResult(data.source_url, data.output_url, {
            toolName: 'Fast-Marching Telea Inpaint',
            modeName: 'Seamless Object Erasure',
            elapsed: data.elapsed,
            likeness: 'Surrounding Texture Matched',
            tag: 'Object Erased'
          });
        } else {
          endProgress();
          canvasPlaceholder.classList.remove('hidden');
          alert("Erase error: " + (data.error || "Failed to erase object"));
        }
      } catch (e) {
        endProgress();
        canvasPlaceholder.classList.remove('hidden');
        alert("Network error: " + e.message);
      } finally {
        eraseBtn.disabled = false;
      }
    });
  }

  // 5. 4K Ultra-Upscaler
  if (upscaleBtn) {
    upscaleBtn.addEventListener('click', async () => {
      if (!currentImageId && !currentSourceUrl) {
        alert("Please upload a photo to upscale.");
        return;
      }
      upscaleBtn.disabled = true;
      startProgress("Upscaling to 4K Ultra-HD via RealESRGAN Super-Resolution...");

      try {
        const payload = {
          image_id: currentImageId,
          model_name: upscaleModelSelect ? upscaleModelSelect.value : 'RealESRGAN_x4plus.pth'
        };
        const res = await fetch('/api/upscale', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success && data.output_url) {
          displayResult(data.source_url, data.output_url, {
            toolName: 'RealESRGAN 4x',
            modeName: `${data.width}x${data.height} 4K UHD`,
            elapsed: data.elapsed,
            likeness: 'Super-Resolution Detail',
            tag: '4K Upscaled'
          });
        } else {
          endProgress();
          canvasPlaceholder.classList.remove('hidden');
          alert("Upscale error: " + (data.error || "Failed to upscale photo"));
        }
      } catch (e) {
        endProgress();
        canvasPlaceholder.classList.remove('hidden');
        alert("Network error: " + e.message);
      } finally {
        upscaleBtn.disabled = false;
      }
    });
  }

  // 6. AI Deblur & Sharpener
  if (deblurBtn) {
    deblurBtn.addEventListener('click', async () => {
      if (!currentImageId && !currentSourceUrl) {
        alert("Please upload a photo to deblur.");
        return;
      }
      deblurBtn.disabled = true;
      startProgress("Deblurring Motion & Sharpening Fine Contours...");

      try {
        const payload = {
          image_id: currentImageId,
          sharpness: deblurStrengthSlider ? parseFloat(deblurStrengthSlider.value) : 1.35
        };
        const res = await fetch('/api/deblur', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success && data.output_url) {
          displayResult(data.source_url, data.output_url, {
            toolName: 'Frequency Deconvolution',
            modeName: `Sharpness ${payload.sharpness}`,
            elapsed: data.elapsed,
            likeness: 'Razor Sharp Features',
            tag: 'Deblur Complete'
          });
        } else {
          endProgress();
          canvasPlaceholder.classList.remove('hidden');
          alert("Deblur error: " + (data.error || "Failed to deblur photo"));
        }
      } catch (e) {
        endProgress();
        canvasPlaceholder.classList.remove('hidden');
        alert("Network error: " + e.message);
      } finally {
        deblurBtn.disabled = false;
      }
    });
  }

  // Download High-Res
  downloadBtn.addEventListener('click', () => {
    if (!currentOutputUrl) return;
    const a = document.createElement('a');
    a.href = currentOutputUrl;
    a.download = `facestudio_result_${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  });

  // -------------------------------------------------------------------------
  // 9. Interactive Before/After Split Slider
  // -------------------------------------------------------------------------
  function updateSliderPosition(percent) {
    const clamped = Math.max(0, Math.min(100, percent));
    splitSliderWrapper.style.setProperty('--split-pos', `${clamped}%`);
    sliderDivider.style.left = `${clamped}%`;
  }

  function handleSliderMove(e) {
    if (!isDraggingSlider) return;
    const rect = splitSliderWrapper.getBoundingClientRect();
    const clientX = (e.touches && e.touches[0]) ? e.touches[0].clientX : e.clientX;
    if (clientX !== undefined && rect.width > 0) {
      const x = clientX - rect.left;
      const percent = (x / rect.width) * 100;
      updateSliderPosition(percent);
    }
  }

  splitSliderWrapper.addEventListener('mousedown', (e) => {
    isDraggingSlider = true;
    handleSliderMove(e);
  });
  window.addEventListener('mousemove', handleSliderMove);
  window.addEventListener('mouseup', () => { isDraggingSlider = false; });

  splitSliderWrapper.addEventListener('touchstart', (e) => {
    isDraggingSlider = true;
    handleSliderMove(e);
  });
  window.addEventListener('touchmove', handleSliderMove);
  window.addEventListener('touchend', () => { isDraggingSlider = false; });
});
