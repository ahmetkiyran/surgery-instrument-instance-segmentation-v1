/** Convert a pointer in an object-fit:contain viewport to source-normalized coordinates.
 * Returns null for black letterbox/pillarbox space, which must never trigger SAM3.
 */
export function normalizeContainedPoint(
  x: number, y: number, viewportWidth: number, viewportHeight: number, sourceWidth: number, sourceHeight: number,
): { normalized_x: number; normalized_y: number } | null {
  if (![x, y, viewportWidth, viewportHeight, sourceWidth, sourceHeight].every(Number.isFinite) || viewportWidth <= 0 || viewportHeight <= 0 || sourceWidth <= 0 || sourceHeight <= 0) return null;
  const scale = Math.min(viewportWidth / sourceWidth, viewportHeight / sourceHeight);
  const imageWidth = sourceWidth * scale;
  const imageHeight = sourceHeight * scale;
  const left = (viewportWidth - imageWidth) / 2;
  const top = (viewportHeight - imageHeight) / 2;
  if (x < left || y < top || x > left + imageWidth || y > top + imageHeight) return null;
  return {
    normalized_x: Math.max(0, Math.min(1, (x - left) / imageWidth)),
    normalized_y: Math.max(0, Math.min(1, (y - top) / imageHeight)),
  };
}
