"""
Screenshot capture for Linux desktop environments.

Detects available screenshot tools and provides a unified interface.
Supports: gnome-screenshot, spectacle (KDE), maim, scrot.
"""

import subprocess
import os
import time
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image


class ScreenshotError(Exception):
    """Raised when screenshot capture fails."""

    pass


class ScreenshotCapture:
    """
    Cross-desktop screenshot capture for Linux.

    Automatically detects and uses the first available screenshot tool.
    Priority order: gnome-screenshot → spectacle → maim → scrot

    Usage:
        capture = ScreenshotCapture()
        image = capture.capture_area()  # Launches area selection
    """

    # Tool name -> command template
    # {output} will be replaced with the output path
    TOOLS: dict[str, List[str]] = {
        "gnome-screenshot": ["gnome-screenshot", "-a", "-f", "{output}"],
        "spectacle": ["spectacle", "-r", "-b", "-n", "-o", "{output}"],
        "maim": ["maim", "-s", "{output}"],
        "scrot": ["scrot", "-s", "{output}"],
    }

    DEFAULT_OUTPUT = "/tmp/mathsolver_capture.png"

    def __init__(self, preferred_tool: Optional[str] = None):
        """
        Initialize screenshot capture.

        Args:
            preferred_tool: Override auto-detection with specific tool name.
        """
        if preferred_tool:
            if preferred_tool not in self.TOOLS:
                raise ValueError(
                    f"Unknown tool: {preferred_tool}. "
                    f"Available: {list(self.TOOLS.keys())}"
                )
            if not self._tool_available(preferred_tool):
                raise ScreenshotError(
                    f"Requested tool '{preferred_tool}' not found in PATH"
                )
            self.tool = preferred_tool
        else:
            detected = self._detect_available_tool()
            if detected is None:
                available = list(self.TOOLS.keys())
                raise ScreenshotError(
                    f"No screenshot tool found. Install one of: {available}\n"
                    "  Debian/Ubuntu: sudo apt install gnome-screenshot\n"
                    "  Arch: sudo pacman -S gnome-screenshot\n"
                    "  Fedora: sudo dnf install gnome-screenshot"
                )
            self.tool = detected

        # Type narrowing: at this point self.tool is definitely str
        if self.tool is None:  # This is now unreachable but helps type checker
            available = list(self.TOOLS.keys())
            raise ScreenshotError(
                f"No screenshot tool found. Install one of: {available}\n"
                "  Debian/Ubuntu: sudo apt install gnome-screenshot\n"
                "  Arch: sudo pacman -S gnome-screenshot\n"
                "  Fedora: sudo dnf install gnome-screenshot"
            )

    def _tool_available(self, tool_name: str) -> bool:
        """Check if a tool is available in PATH."""
        result = subprocess.run(["which", tool_name], capture_output=True, text=True)
        return result.returncode == 0

    def _detect_available_tool(self) -> Optional[str]:
        """Detect first available screenshot tool."""
        for tool_name in self.TOOLS.keys():
            if self._tool_available(tool_name):
                return tool_name
        return None

    def capture_area(self, output_path: Optional[str] = None) -> Image.Image:
        """
        Launch area selection and capture screenshot.

        Args:
            output_path: Where to save the screenshot. Defaults to /tmp.

        Returns:
            PIL Image object of the captured region.

        Raises:
            ScreenshotError: If capture fails or user cancels.
        """
        output_path = output_path or self.DEFAULT_OUTPUT

        # Clean up old file if exists
        if os.path.exists(output_path):
            os.remove(output_path)

        # Build command with output path
        cmd = [arg.replace("{output}", output_path) for arg in self.TOOLS[self.tool]]

        # Execute screenshot tool
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check for errors
        if result.returncode != 0:
            # User cancelled is usually returncode 1 with no stderr
            if result.returncode == 1 and not result.stderr:
                raise ScreenshotError("Screenshot cancelled by user")
            raise ScreenshotError(
                f"Screenshot failed (exit code {result.returncode}): {result.stderr}"
            )

        # Verify file was created (some tools exit 0 even on cancel)
        if not os.path.exists(output_path):
            raise ScreenshotError("Screenshot was not saved (possibly cancelled)")

        # Load and return image
        try:
            image = Image.open(output_path)
            # Convert to RGB if necessary (some screenshots are RGBA)
            if image.mode == "RGBA":
                image = image.convert("RGB")
            return image
        except Exception as e:
            raise ScreenshotError(f"Failed to load screenshot: {e}")

    def capture_fullscreen(self, output_path: Optional[str] = None) -> Image.Image:
        """
        Capture the full screen (no selection).

        Note: Not all tools support this equally well.
        """
        # For now, just do area selection
        # TODO: Add fullscreen-specific commands
        return self.capture_area(output_path)

    @property
    def tool_name(self) -> str:
        """Return the name of the screenshot tool being used."""
        # self.tool is guaranteed to be str after __init__ (raises if None)
        assert self.tool is not None
        return self.tool


def get_available_tools() -> List[str]:
    """Return list of screenshot tools available on this system."""
    capture = ScreenshotCapture.__new__(ScreenshotCapture)
    return [
        tool for tool in ScreenshotCapture.TOOLS.keys() if capture._tool_available(tool)
    ]
