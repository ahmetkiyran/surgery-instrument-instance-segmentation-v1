import { useEffect, useState } from "react";
import { AccessibilityInfo, Text, View } from "react-native";
import LottieView from "lottie-react-native";
import idle from "../../desktop/assets/lottie/idle.json";
import checkingModels from "../../desktop/assets/lottie/checking_models.json";
import uploading from "../../desktop/assets/lottie/uploading.json";
import analyzing from "../../desktop/assets/lottie/analyzing.json";
import completed from "../../desktop/assets/lottie/completed.json";
import error from "../../desktop/assets/lottie/error.json";

const animations = { idle, checking_models: checkingModels, uploading, analyzing, completed, error };
export type AnimationState = keyof typeof animations;

export function StatusAnimation({ state }: { state: AnimationState }) {
  const [reduceMotion, setReduceMotion] = useState(false);
  useEffect(() => {
    void AccessibilityInfo.isReduceMotionEnabled().then(setReduceMotion);
    const subscription = AccessibilityInfo.addEventListener("reduceMotionChanged", setReduceMotion);
    return () => subscription.remove();
  }, []);
  if (reduceMotion) return <View accessibilityLabel={state}><Text style={{ color: "#1595aa", fontSize: 28 }}>●</Text></View>;
  return <LottieView source={animations[state]} autoPlay loop={state === "checking_models" || state === "uploading" || state === "analyzing"} style={{ width: 56, height: 56 }} />;
}
