/** Minimal inline icons in the SF Symbols idiom — stroked, 1.6px, round caps. */

const base = {
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

export const GearIcon = () => (
  <svg {...base} width="20" height="20">
    <circle cx="12" cy="12" r="3.2" />
    <path d="M19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-1.8-.3 1.6 1.6 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1A1.6 1.6 0 0 0 9 19.4a1.6 1.6 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0 .3-1.8 1.6 1.6 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1A1.6 1.6 0 0 0 4.6 9a1.6 1.6 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3H9a1.6 1.6 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 1 1.5 1.6 1.6 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8V9a1.6 1.6 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1Z" />
  </svg>
);

export const MicIcon = () => (
  <svg {...base} width="19" height="19">
    <rect x="9" y="2.5" width="6" height="11" rx="3" />
    <path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21" />
  </svg>
);

export const MicOffIcon = () => (
  <svg {...base} width="19" height="19">
    <path d="M9 5.5a3 3 0 0 1 6 0V11m-6 .5V11" />
    <path d="M5.5 11a6.5 6.5 0 0 0 10.2 5.3M18.5 11v0M12 17.5V21" />
    <path d="M3.5 3.5l17 17" />
  </svg>
);

export const CloseIcon = () => (
  <svg {...base} width="20" height="20">
    <path d="M6 6l12 12M18 6L6 18" />
  </svg>
);

export const SendIcon = () => (
  <svg {...base} width="18" height="18" strokeWidth={2}>
    <path d="M12 19V5M5 12l7-7 7 7" />
  </svg>
);

export const ChatIcon = () => (
  <svg {...base} width="19" height="19">
    <path d="M20.5 11.5a7.5 7.5 0 0 1-7.5 7.5H8l-4 2.5V19a7.5 7.5 0 0 1 4-14h1a7.5 7.5 0 0 1 7.5 6.5Z" />
    <path d="M8.5 11.5h7M8.5 8h5" />
  </svg>
);

export const WaveIcon = () => (
  <svg {...base} width="19" height="19">
    <path d="M4 10v4M8 7v10M12 4.5v15M16 8v8M20 10.5v3" />
  </svg>
);
