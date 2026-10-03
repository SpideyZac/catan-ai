import { setMusicOn, useMusicOn } from "../lib/music";
import { setMuted, useMuted } from "../lib/sound";

/** Speaker button: mutes / unmutes sound effects. */
export function SoundToggle() {
  const muted = useMuted();
  const label = muted ? "Unmute sound effects" : "Mute sound effects";
  return (
    <button
      className="btn ghost small sound-toggle"
      onClick={() => setMuted(!muted)}
      title={label}
      aria-label={label}
      aria-pressed={muted}
    >
      {muted ? "🔇" : "🔊"}
    </button>
  );
}

/** Note button: turns the background music on / off. */
export function MusicToggle() {
  const on = useMusicOn();
  const label = on ? "Turn music off" : "Turn music on";
  return (
    <button
      className={`btn ghost small sound-toggle music-toggle ${on ? "" : "off"}`}
      onClick={() => setMusicOn(!on)}
      title={label}
      aria-label={label}
      aria-pressed={on}
    >
      🎵
    </button>
  );
}
