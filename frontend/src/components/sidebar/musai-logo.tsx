'use client';

import Image from 'next/image';
import { useTheme } from 'next-themes';
import { useEffect, useState } from 'react';

interface MusaiLogoProps {
  size?: number;
}
export function MusaiLogo({ size = 24 }: MusaiLogoProps) {
  const { theme, systemTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // After mount, we can access the theme
  useEffect(() => {
    setMounted(true);
  }, []);

  const isDark = mounted && (
    theme === 'dark' || (theme === 'system' && systemTheme === 'dark')
  );

  return (
    <Image
        src={isDark ? "/wtMusai.png" : "/Musai.png"}
        alt="Musai"
        width={size}
        height={size}
        className="flex-shrink-0"
      />
  );
} 