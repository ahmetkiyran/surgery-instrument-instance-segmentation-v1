import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles.css";

// The managed Python API is a process-owned resource. React StrictMode's
// development-only mount/unmount probe would start it, stop it, and race the
// second mount, leaving the UI with a stale loopback URL.
createRoot(document.getElementById("root")!).render(<App />);
