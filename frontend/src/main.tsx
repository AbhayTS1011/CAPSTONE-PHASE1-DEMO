import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App";
import ComponentPreview from "./components/demo/component-preview";

// "#/components" shows the integrated UI primitives in isolation (development preview).
const Root = window.location.hash.startsWith("#/components") ? ComponentPreview : App;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
);
