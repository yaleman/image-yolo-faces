import "./styles.css";
import { bindUploadRoots } from "./upload";

function bindConfirmForms(): void {
  for (const form of document.querySelectorAll<HTMLFormElement>(
    "form[data-confirm]",
  )) {
    form.addEventListener("submit", (event) => {
      const message = form.dataset.confirm;
      if (message && !window.confirm(message)) {
        event.preventDefault();
      }
    });
  }
}

bindConfirmForms();
bindUploadRoots();
