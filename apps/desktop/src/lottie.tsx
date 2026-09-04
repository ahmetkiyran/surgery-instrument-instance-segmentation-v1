import { Lottie } from "lottie-react";
import { useEffect, useState } from "react";
import idle from "../assets/lottie/idle.json";
import checkingModels from "../assets/lottie/checking_models.json";
import uploading from "../assets/lottie/uploading.json";
import analyzing from "../assets/lottie/analyzing.json";
import completed from "../assets/lottie/completed.json";
import error from "../assets/lottie/error.json";

const animations = { idle, checking_models: checkingModels, uploading, analyzing, completed, error };
export type AnimationState = keyof typeof animations;

export function StatusAnimation({ state }: { state: AnimationState }) {
  const [reduceMotion, setReduceMotion] = useState(false);
  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduceMotion(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  if (reduceMotion) return <span className="static-status-icon" aria-label={state}>●</span>;
  return <Lottie className="status-animation" src={animations[state]} autoplay loop={state === "analyzing" || state === "checking_models" || state === "uploading"} aria-label={state} />;
}
