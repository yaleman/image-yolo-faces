type UploadStatus = "duplicate" | "imported";

type UploadPayload = {
  detail?: string;
  detail_url?: string;
  image_id?: string;
  status?: UploadStatus;
};

type UploadElements = {
  disclosure: HTMLDetailsElement | null;
  feedback: HTMLElement;
  filename: HTMLElement;
  form: HTMLFormElement;
  input: HTMLInputElement;
  overlayBody: HTMLElement;
  overlay: HTMLElement;
  overlayTitle: HTMLElement;
  submit: HTMLButtonElement;
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

function setFilename(filenameNode: HTMLElement, file?: File): void {
  filenameNode.textContent = file ? file.name : "No file selected";
}

function assignFile(input: HTMLInputElement, file: File): void {
  const dataTransfer = new DataTransfer();
  dataTransfer.items.add(file);
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

  if (
    !form ||
    !input ||
    !feedback ||
    !filename ||
    !submit ||
    !overlay ||
    !overlayTitle ||
    !overlayBody
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
    submit,
  };
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
    submit,
  } = elements;
  let dragDepth = 0;
  let keepOverlayVisible = false;

  const setBusy = (busy: boolean): void => {
    submit.disabled = busy;
    submit.textContent = busy ? "Uploading..." : "Upload image";
    overlayTitle.textContent = busy
      ? "Processing..."
      : "Drop one image anywhere to import it";
    overlayBody.textContent = busy
      ? "Uploading the image, checking its hash, and scanning for faces."
      : "We will check the SHA-256 hash, reuse an existing record if it already exists, or scan and add it if it is new.";
  };

  const handleFile = (file?: File): void => {
    if (!file) {
      return;
    }

    assignFile(input, file);
    setFilename(filename, file);
    clearFeedback(feedback);
  };

  const submitFile = async (
    file: File,
    options: { keepOverlay?: boolean } = {},
  ): Promise<void> => {
    keepOverlayVisible = options.keepOverlay === true;
    handleFile(file);
    const formData = new FormData();
    formData.set("image", file, file.name);
    setBusy(true);
    if (keepOverlayVisible) {
      root.classList.add("is-uploading");
      showOverlay(root, overlay);
    }

    try {
      const response = await fetch(form.action, {
        body: formData,
        method: "POST",
      });
      const payload = (await response.json()) as UploadPayload;

      if (!response.ok) {
        if (disclosure) {
          disclosure.open = true;
        }
        hideOverlay(root, overlay);
        setFeedback(
          feedback,
          "error",
          payload.detail || "Upload failed. Please try again.",
        );
        return;
      }

      const label =
        payload.status === "duplicate"
          ? "That image already exists."
          : "Image imported and scanned.";
      setFeedback(
        feedback,
        "success",
        `${label} <a class="subtle-link" href="${payload.detail_url}">Open image</a>`,
      );

      const detailUrl = payload.detail_url;
      if (detailUrl) {
        window.setTimeout(() => {
          window.location.assign(detailUrl);
        }, 500);
      } else {
        hideOverlay(root, overlay);
      }
    } catch (error) {
      console.error(error);
      if (disclosure) {
        disclosure.open = true;
      }
      hideOverlay(root, overlay);
      setFeedback(feedback, "error", "Upload failed. Please try again.");
    } finally {
      setBusy(false);
      if (!keepOverlayVisible) {
        hideOverlay(root, overlay);
      }
      keepOverlayVisible = false;
    }
  };

  input.addEventListener("change", () => {
    setFilename(filename, input.files?.[0]);
    clearFeedback(feedback);
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
    if (root.classList.contains("is-uploading")) {
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
    if (!root.classList.contains("is-uploading")) {
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
    const [file] = event.dataTransfer?.files ?? [];
    if (!file) {
      hideOverlay(root, overlay);
      return;
    }
    void submitFile(file, { keepOverlay: true });
  });

  overlay.addEventListener("click", () => {
    input.click();
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const [file] = input.files ?? [];
    if (!file) {
      setFeedback(feedback, "error", "Choose an image before uploading.");
      return;
    }
    await submitFile(file);
  });
}

export function bindUploadRoots(): void {
  for (const root of document.querySelectorAll<HTMLElement>(
    "[data-upload-root]",
  )) {
    bindUploadRoot(root);
  }
}
