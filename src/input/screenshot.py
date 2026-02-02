"""
Screenshot capture for Linux desktop environments.

Detects available screenshot tools and provides a unified interface.
Supports: gnome-screenshot, spectacle (KDE), maim, scrot, flameshot.
"""

import subprocess
import os
import time
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image

from ..utils.errors import ScreenshotError, ScreenshotCancelledError


class ScreenshotCapture:
    """
    Cross-desktop screenshot capture for Linux.

    Automatically detects and uses the first available screenshot tool.
    Priority order: flameshot → gnome-screenshot → spectacle → maim → scrot

    Usage:
        capture = ScreenshotCapture()
        image = capture.capture_area()  # Launches area selection
    """

    # Tool name -> command template
    # {output} will be replaced with the output path
    TOOLS: dict[str, List[str]] = {
        "flameshot": ["flameshot", "gui", "--raw", "-p", "{output}"],
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
            ScreenshotCancelledError: If user cancels the selection.
            ScreenshotError: If capture fails for other reasons.
        """
        output_path = output_path or self.DEFAULT_OUTPUT

        # Clean up old file if exists
        if os.path.exists(output_path):
            os.remove(output_path)

        # Build command with output path
        cmd = [arg.replace("{output}", output_path) for arg in self.TOOLS[self.tool]]

        # Execute screenshot tool
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            raise ScreenshotError(
                "Screenshot tool timed out after 60 seconds",
                suggestions=["Try again and select an area more quickly"],
            )

        # Check for errors
        if result.returncode != 0:
            # User cancelled is usually returncode 1 with no stderr
            # or specific exit codes for different tools
            stderr_lower = result.stderr.lower() if result.stderr else ""

            if result.returncode == 1 and not result.stderr:
                raise ScreenshotCancelledError()
            if "cancel" in stderr_lower or "aborted" in stderr_lower:
                raise ScreenshotCancelledError()

            raise ScreenshotError(
                f"Screenshot failed (exit code {result.returncode})",
                technical_details=result.stderr if result.stderr else None,
            )

        # Verify file was created (some tools exit 0 even on cancel)
        if not os.path.exists(output_path):
            raise ScreenshotCancelledError()

        # Check file size (empty file means cancelled for some tools)
        if os.path.getsize(output_path) == 0:
            os.remove(output_path)
            raise ScreenshotCancelledError()

        # Load and return image
        try:
            image = Image.open(output_path)
            # Convert to RGB if necessary (some screenshots are RGBA)
            if image.mode == "RGBA":
                image = image.convert("RGB")
            return image
        except Exception as e:
            raise ScreenshotError(
                f"Failed to load screenshot image",
                technical_details=str(e),
                suggestions=[
                    "The captured image may be corrupted",
                    "Try capturing again",
                ],
            )

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
