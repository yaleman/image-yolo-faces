type UploadStatus = "duplicate" | "error" | "imported";

type UploadResult = {
  detail?: string;
  detail_url?: string;
  filename?: string;
  image_id?: string;
  status?: UploadStatus;
};

type UploadResponse = {
  detail?: string;
  results?: UploadResult[];
  status?: UploadStatus;
  detail_url?: string;
  filename?: string;
  image_id?: string;
};

type UploadElements = {
  disclosure: HTMLDetailsElement | null;
  feedback: HTMLElement;
  filename: HTMLElement;
  form: HTMLFormElement;
  input: HTMLInputElement;
  overlay: HTMLElement;
  overlayBody: HTMLElement;
  overlayTitle: HTMLElement;
  status: HTMLElement;
  statusList: HTMLOListElement;
  statusSummary: HTMLElement;
  submit: HTMLButtonElement;
};

type UploadStatusKind =
  | "queued"
  | "uploading"
  | "processing"
  | "success"
  | "duplicate"
  | "error";

type UploadStatusItem = {
  file: File;
  item: HTMLLIElement;
  progress: HTMLProgressElement;
  state: HTMLElement;
  link: HTMLAnchorElement;
};

function isFileDrag(event: DragEvent): boolean {
  return Array.from(event.dataTransfer?.types ?? []).includes("Files");
}

function setFeedback(
  feedbackNode: HTMLElement,
  kind: "error" | "success",
  message: string,
): void {
  feedbackNode.classList.remove("hidden", "is-error", "is-success");
  feedbackNode.classList.add(kind === "error" ? "is-error" : "is-success");
  feedbackNode.innerHTML = message;
}

function clearFeedback(feedbackNode: HTMLElement): void {
  feedbackNode.classList.add("hidden");
  feedbackNode.innerHTML = "";
}

function setFilenameSummary(filenameNode: HTMLElement, files: File[]): void {
  if (files.length === 0) {
    filenameNode.textContent = "No files selected";
    return;
  }

  if (files.length === 1) {
    filenameNode.textContent = files[0].name;
    return;
  }

  const previewNames = files.slice(0, 2).map((file) => file.name);
  const remainingCount = files.length - previewNames.length;
  filenameNode.textContent =
    remainingCount > 0
      ? `${previewNames.join(", ")} +${remainingCount} more`
      : previewNames.join(", ");
}

function assignFiles(input: HTMLInputElement, files: File[]): void {
  const dataTransfer = new DataTransfer();
  for (const file of files) {
    dataTransfer.items.add(file);
  }
  input.files = dataTransfer.files;
}

function hideOverlay(root: HTMLElement, overlay: HTMLElement): void {
  root.classList.remove("is-drag-over");
  root.classList.remove("is-uploading");
  overlay.classList.add("hidden");
}

function showOverlay(root: HTMLElement, overlay: HTMLElement): void {
  root.classList.add("is-drag-over");
  overlay.classList.remove("hidden");
}

function getUploadElements(root: Element): UploadElements | null {
  const form = root.querySelector<HTMLFormElement>("[data-upload-form]");
  const input = root.querySelector<HTMLInputElement>("[data-upload-input]");
  const feedback = root.querySelector<HTMLElement>("[data-upload-feedback]");
  const filename = root.querySelector<HTMLElement>("[data-upload-filename]");
  const submit = root.querySelector<HTMLButtonElement>("[data-upload-submit]");
  const overlay = root.querySelector<HTMLElement>("[data-upload-overlay]");
  const overlayTitle = root.querySelector<HTMLElement>(
    "[data-upload-overlay-title]",
  );
  const overlayBody = root.querySelector<HTMLElement>(
    "[data-upload-overlay-body]",
  );
  const disclosure = root.querySelector<HTMLDetailsElement>(
    "[data-upload-disclosure]",
  );
  const status = root.querySelector<HTMLElement>("[data-upload-status]");
  const statusSummary = root.querySelector<HTMLElement>(
    "[data-upload-status-summary]",
  );
  const statusList = root.querySelector<HTMLOListElement>(
    "[data-upload-status-list]",
  );

  if (
    !form ||
    !input ||
    !feedback ||
    !filename ||
    !submit ||
    !overlay ||
    !overlayTitle ||
    !overlayBody ||
    !status ||
    !statusSummary ||
    !statusList
  ) {
    return null;
  }

  return {
    disclosure,
    feedback,
    filename,
    form,
    input,
    overlay,
    overlayBody,
    overlayTitle,
    status,
    statusList,
    statusSummary,
    submit,
  };
}

function clearStatusList(
  status: HTMLElement,
  statusList: HTMLOListElement,
): void {
  statusList.replaceChildren();
  status.classList.add("hidden");
}

function setStatusSummary(
  status: HTMLElement,
  statusSummary: HTMLElement,
  message: string,
): void {
  statusSummary.textContent = message;
  status.classList.remove("hidden");
}

function createStatusItem(file: File): UploadStatusItem {
  const item = document.createElement("li");
  item.className = "upload-status-item is-queued";

  const row = document.createElement("div");
  row.className = "upload-status-item-row";

  const name = document.createElement("span");
  name.className = "upload-status-name";
  name.textContent = file.name;

  const state = document.createElement("span");
  state.className = "upload-status-state";
  state.textContent = "Queued";

  const link = document.createElement("a");
  link.className = "upload-status-link hidden";
  link.target = "_blank";
  link.rel = "noreferrer";
  link.textContent = "Open image";

  const progress = document.createElement("progress");
  progress.className = "upload-status-progress";
  progress.max = 100;
  progress.value = 0;

  row.append(name, state, link);
  item.append(row, progress);

  return {
    file,
    item,
    progress,
    state,
    link,
  };
}

function setItemState(
  item: UploadStatusItem,
  state: UploadStatusKind,
  message: string,
): void {
  item.item.classList.remove(
    "is-queued",
    "is-uploading",
    "is-processing",
    "is-success",
    "is-duplicate",
    "is-error",
  );
  item.item.classList.add(`is-${state}`);
  item.state.textContent = message;
}

function setItemProgress(
  item: UploadStatusItem,
  loaded: number,
  total: number,
): void {
  const percent =
    total > 0 ? Math.min(100, Math.max(0, (loaded / total) * 100)) : 0;
  item.progress.value = percent;
}

function finalizeItem(
  item: UploadStatusItem,
  result: UploadResult,
  fallbackStatus: UploadStatusKind = "success",
): void {
  const status = result.status ?? "imported";
  const finalState =
    status === "duplicate"
      ? "duplicate"
      : status === "error"
        ? "error"
        : fallbackStatus;

  item.progress.value = 100;

  if (finalState === "duplicate") {
    setItemState(item, finalState, result.detail || "Already imported.");
  } else if (finalState === "error") {
    setItemState(item, finalState, result.detail || "Upload failed.");
  } else {
    setItemState(item, finalState, "Imported and scanned.");
  }

  if (result.detail_url) {
    item.link.href = result.detail_url;
    item.link.classList.remove("hidden");
  }
}

function parseUploadResponse(responseText: string): UploadResponse | null {
  try {
    return JSON.parse(responseText) as UploadResponse;
  } catch {
    return null;
  }
}

function normalizeUploadResult(payload: UploadResponse): UploadResult | null {
  if (Array.isArray(payload.results) && payload.results.length > 0) {
    return payload.results[0];
  }

  if (
    payload.status ||
    payload.detail ||
    payload.detail_url ||
    payload.filename ||
    payload.image_id
  ) {
    return payload;
  }

  return null;
}

function extractErrorMessage(responseText: string): string {
  const payload = parseUploadResponse(responseText);
  if (payload && typeof payload.detail === "string" && payload.detail.trim()) {
    return payload.detail;
  }
  return "Upload failed.";
}

function uploadFile(
  action: string,
  file: File,
  onProgress: (loaded: number, total: number) => void,
  onPhase: (phase: "uploading" | "processing") => void,
): Promise<UploadResult> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", action);
    request.responseType = "text";

    request.upload.addEventListener("loadstart", () => {
      onPhase("uploading");
    });

    request.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable) {
        return;
      }

      onPhase(event.loaded < event.total ? "uploading" : "processing");
      onProgress(event.loaded, event.total);
    });

    request.upload.addEventListener("load", () => {
      onPhase("processing");
    });

    request.addEventListener("load", () => {
      const payload = parseUploadResponse(request.responseText);
      if (!payload) {
        reject(new Error("Upload failed."));
        return;
      }

      if (request.status < 200 || request.status >= 300) {
        reject(new Error(extractErrorMessage(request.responseText)));
        return;
      }

      const result = normalizeUploadResult(payload);
      if (!result) {
        reject(new Error("Upload failed."));
        return;
      }

      resolve(result);
    });

    request.addEventListener("error", () => {
      reject(new Error("Upload failed."));
    });

    request.addEventListener("abort", () => {
      reject(new Error("Upload canceled."));
    });

    const formData = new FormData();
    formData.append("image", file, file.name);
    request.send(formData);
  });
}

function collectFiles(input: HTMLInputElement): File[] {
  return Array.from(input.files ?? []);
}

function bindUploadRoot(root: HTMLElement): void {
  const elements = getUploadElements(root);
  if (!elements) {
    return;
  }

  const {
    disclosure,
    feedback,
    filename,
    form,
    input,
    overlay,
    overlayBody,
    overlayTitle,
    status,
    statusList,
    statusSummary,
    submit,
  } = elements;

  let dragDepth = 0;
  let uploadInProgress = false;

  const setBusy = (busy: boolean): void => {
    submit.disabled = busy;
    submit.textContent = busy ? "Uploading..." : "Upload images";
    overlayTitle.textContent = busy
      ? "Processing uploads..."
      : "Drop images anywhere to import them";
    overlayBody.textContent = busy
      ? "Uploading each image, checking its hash, and scanning for faces."
      : "We will check the SHA-256 hash, reuse an existing record if it already exists, or scan and add it if it is new.";
  };

  const clearQueue = (): void => {
    clearStatusList(status, statusList);
    statusSummary.textContent = "";
  };

  const prepareQueue = (files: File[]): UploadStatusItem[] => {
    clearQueue();
    const items = files.map((file) => createStatusItem(file));
    for (const item of items) {
      statusList.appendChild(item.item);
    }
    setStatusSummary(
      status,
      statusSummary,
      `Processing ${files.length} file${files.length === 1 ? "" : "s"}.`,
    );
    return items;
  };

  const uploadQueue = async (files: File[]): Promise<void> => {
    if (uploadInProgress) {
      return;
    }

    if (files.length === 0) {
      setFeedback(
        feedback,
        "error",
        "Choose at least one image before uploading.",
      );
      return;
    }

    uploadInProgress = true;
    setBusy(true);
    root.classList.add("is-uploading");
    showOverlay(root, overlay);
    clearFeedback(feedback);

    const items = prepareQueue(files);
    let importedCount = 0;
    let duplicateCount = 0;
    let errorCount = 0;
    let redirectUrl: string | null = null;

    try {
      for (const [index, file] of files.entries()) {
        const item = items[index];
        setStatusSummary(
          status,
          statusSummary,
          `Processing ${index + 1} of ${files.length}: ${file.name}`,
        );
        setItemState(item, "uploading", "Uploading...");
        item.progress.value = 0;

        try {
          const result = await uploadFile(
            form.action,
            file,
            (loaded, total) => {
              setItemProgress(item, loaded, total);
              if (loaded >= total) {
                setItemState(item, "processing", "Processing image...");
              } else {
                const percent = Math.round((loaded / total) * 100);
                setItemState(item, "uploading", `Uploading ${percent}%`);
              }
            },
            (phase) => {
              setItemState(
                item,
                phase === "uploading" ? "uploading" : "processing",
                phase === "uploading" ? "Uploading..." : "Processing image...",
              );
            },
          );

          finalizeItem(item, result);

          if (result.status === "duplicate") {
            duplicateCount += 1;
          } else if (result.status === "error") {
            errorCount += 1;
          } else {
            importedCount += 1;
          }

          if (files.length === 1 && result.detail_url) {
            redirectUrl = result.detail_url;
          }
        } catch (error) {
          errorCount += 1;
          const message =
            error instanceof Error ? error.message : "Upload failed.";
          setItemState(item, "error", message);
          item.progress.value = 100;
        }
      }

      const summaryParts: string[] = [];
      if (importedCount > 0) {
        summaryParts.push(`${importedCount} imported`);
      }
      if (duplicateCount > 0) {
        summaryParts.push(
          `${duplicateCount} duplicate${duplicateCount === 1 ? "" : "s"}`,
        );
      }
      if (errorCount > 0) {
        summaryParts.push(`${errorCount} failed`);
      }

      if (summaryParts.length === 0) {
        summaryParts.push("No images were processed.");
      }

      setStatusSummary(
        status,
        statusSummary,
        `Finished processing ${files.length} file${files.length === 1 ? "" : "s"}.`,
      );

      const kind = errorCount > 0 ? "error" : "success";
      setFeedback(feedback, kind, `${summaryParts.join(" • ")}.`);

      if (redirectUrl) {
        window.setTimeout(() => {
          window.location.assign(redirectUrl ?? "/");
        }, 500);
      }
    } catch (error) {
      console.error(error);
      if (disclosure) {
        disclosure.open = true;
      }
      setFeedback(feedback, "error", "Upload failed. Please try again.");
    } finally {
      uploadInProgress = false;
      setBusy(false);
      if (!redirectUrl) {
        root.classList.remove("is-uploading");
        hideOverlay(root, overlay);
      }
    }
  };

  const applyFiles = (files: File[], keepOverlay = false): void => {
    assignFiles(input, files);
    setFilenameSummary(filename, files);
    clearFeedback(feedback);
    clearQueue();

    if (keepOverlay) {
      showOverlay(root, overlay);
    }
  };

  input.addEventListener("change", () => {
    applyFiles(collectFiles(input));
  });

  window.addEventListener("dragenter", (event) => {
    if (!isFileDrag(event)) {
      return;
    }
    event.preventDefault();
    dragDepth += 1;
    showOverlay(root, overlay);
  });

  window.addEventListener("dragover", (event) => {
    if (!isFileDrag(event)) {
      return;
    }
    event.preventDefault();
    showOverlay(root, overlay);
  });

  window.addEventListener("dragleave", (event) => {
    if (!isFileDrag(event)) {
      return;
    }
    if (uploadInProgress) {
      return;
    }
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0 && event.relatedTarget === null) {
      hideOverlay(root, overlay);
    }
  });

  window.addEventListener("drop", (event) => {
    if (!isFileDrag(event)) {
      return;
    }
    event.preventDefault();
    dragDepth = 0;
    if (!uploadInProgress) {
      hideOverlay(root, overlay);
    }
  });

  overlay.addEventListener("dragenter", (event) => {
    if (!isFileDrag(event)) {
      return;
    }
    event.preventDefault();
    showOverlay(root, overlay);
  });

  overlay.addEventListener("dragover", (event) => {
    if (!isFileDrag(event)) {
      return;
    }
    event.preventDefault();
    showOverlay(root, overlay);
  });

  overlay.addEventListener("drop", (event) => {
    if (!isFileDrag(event)) {
      return;
    }
    event.preventDefault();
    dragDepth = 0;
    const files = Array.from(event.dataTransfer?.files ?? []);
    if (files.length === 0) {
      hideOverlay(root, overlay);
      return;
    }
    applyFiles(files, true);
    void uploadQueue(files);
  });

  overlay.addEventListener("click", () => {
    input.click();
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const files = collectFiles(input);
    applyFiles(files);
    await uploadQueue(files);
  });
}

export function bindUploadRoots(): void {
  for (const root of document.querySelectorAll<HTMLElement>(
    "[data-upload-root]",
  )) {
    bindUploadRoot(root);
  }
}
