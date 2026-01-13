"""
Integrated minimap calibration - shows UI when needed during service startup
"""

import tkinter as tk
from typing import Optional, Tuple


class MinimapCalibrationOverlay:
    """
    Semi-transparent overlay for calibrating minimap region
    Runs synchronously and blocks until user completes calibration
    """

    def __init__(self, default_x: int = 0, default_y: int = 0, default_size: int = 260):
        self.result: Optional[Tuple[int, int, int, int]] = None
        self.locked = False
        self.resize_margin = 10
        self.is_resizing = False
        self.resize_direction = None

        # Create window
        self.root = tk.Tk()
        self.root.title("Minimap Calibration")

        # Calculate default position (bottom-right)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        default_x = default_x if default_x > 0 else screen_width - default_size - 20
        default_y = default_y if default_y > 0 else screen_height - default_size - 60

        self.root.geometry(f"{default_size}x{default_size}+{default_x}+{default_y}")
        self.root.attributes('-alpha', 0.5)
        self.root.attributes('-topmost', True)
        self.root.overrideredirect(True)

        self._create_ui()
        self._bind_events()

    def _create_ui(self):
        """Create calibration UI"""
        self.main_frame = tk.Frame(self.root, bg='#00ff00', highlightbackground='#00ff00', highlightthickness=3)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Control panel
        control_frame = tk.Frame(self.main_frame, bg='#1a1a1a', height=40)
        control_frame.pack(fill=tk.X, side=tk.TOP)
        control_frame.pack_propagate(False)

        self.lock_btn = tk.Button(
            control_frame,
            text="🔓 UNLOCK - Drag to Position",
            command=self._toggle_lock,
            bg='#ffaa00',
            fg='#000000',
            font=('Arial', 9, 'bold'),
            relief=tk.FLAT,
            cursor='hand2'
        )
        self.lock_btn.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        self.save_btn = tk.Button(
            control_frame,
            text="✓ SAVE",
            command=self._save,
            bg='#00ff88',
            fg='#000000',
            font=('Arial', 9, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Info
        info_frame = tk.Frame(self.main_frame, bg='#2a2a2a')
        info_frame.pack(fill=tk.BOTH, expand=True)

        self.info_label = tk.Label(
            info_frame,
            text="⚠ MINIMAP CALIBRATION REQUIRED\n\n"
                 "Drag this over your minimap\n"
                 "Resize from edges to fit\n\n"
                 "Click LOCK → SAVE when done",
            bg='#2a2a2a',
            fg='#00ff00',
            font=('Arial', 10),
            justify=tk.CENTER
        )
        self.info_label.pack(expand=True)

        # Position display
        self.pos_label = tk.Label(
            self.main_frame,
            text="Position: (0, 0) | Size: 260x260",
            bg='#1a1a1a',
            fg='#cccccc',
            font=('Courier', 8),
            padx=5,
            pady=3
        )
        self.pos_label.pack(fill=tk.X, side=tk.BOTTOM)

    def _bind_events(self):
        """Bind mouse events"""
        self.main_frame.bind('<Button-1>', self._on_press)
        self.main_frame.bind('<B1-Motion>', self._on_drag)
        self.main_frame.bind('<ButtonRelease-1>', self._on_release)
        self.main_frame.bind('<Motion>', self._on_hover)
        self.info_label.bind('<Button-1>', self._on_press)
        self.info_label.bind('<B1-Motion>', self._on_drag)
        self._update_position_label()

    def _on_hover(self, event):
        """Update cursor based on position"""
        if self.locked:
            self.main_frame.config(cursor='')
            return

        x, y = event.x, event.y
        width = self.root.winfo_width()
        height = self.root.winfo_height()

        on_left = x < self.resize_margin
        on_right = x > width - self.resize_margin
        on_top = y < self.resize_margin
        on_bottom = y > height - self.resize_margin

        if (on_top and on_left) or (on_bottom and on_right):
            self.main_frame.config(cursor='size_nw_se')
        elif (on_top and on_right) or (on_bottom and on_left):
            self.main_frame.config(cursor='size_ne_sw')
        elif on_left or on_right:
            self.main_frame.config(cursor='size_we')
        elif on_top or on_bottom:
            self.main_frame.config(cursor='size_ns')
        else:
            self.main_frame.config(cursor='fleur')

    def _on_press(self, event):
        """Start drag/resize"""
        if self.locked:
            return

        self.start_x = event.x_root
        self.start_y = event.y_root
        self.start_width = self.root.winfo_width()
        self.start_height = self.root.winfo_height()
        self.start_win_x = self.root.winfo_x()
        self.start_win_y = self.root.winfo_y()

        x, y = event.x, event.y
        width = self.root.winfo_width()
        height = self.root.winfo_height()

        on_left = x < self.resize_margin
        on_right = x > width - self.resize_margin
        on_top = y < self.resize_margin
        on_bottom = y > height - self.resize_margin

        if on_left or on_right or on_top or on_bottom:
            self.is_resizing = True
            self.resize_direction = {'left': on_left, 'right': on_right, 'top': on_top, 'bottom': on_bottom}
        else:
            self.is_resizing = False

    def _on_drag(self, event):
        """Handle drag/resize"""
        if self.locked:
            return

        dx = event.x_root - self.start_x
        dy = event.y_root - self.start_y

        if self.is_resizing:
            new_width = self.start_width
            new_height = self.start_height
            new_x = self.start_win_x
            new_y = self.start_win_y

            if self.resize_direction['right']:
                new_width = max(100, self.start_width + dx)
            elif self.resize_direction['left']:
                new_width = max(100, self.start_width - dx)
                new_x = self.start_win_x + dx

            if self.resize_direction['bottom']:
                new_height = max(100, self.start_height + dy)
            elif self.resize_direction['top']:
                new_height = max(100, self.start_height - dy)
                new_y = self.start_win_y + dy

            self.root.geometry(f"{new_width}x{new_height}+{new_x}+{new_y}")
        else:
            new_x = self.start_win_x + dx
            new_y = self.start_win_y + dy
            self.root.geometry(f"+{new_x}+{new_y}")

    def _on_release(self, event):
        """End drag/resize"""
        self.is_resizing = False
        self.resize_direction = None

    def _toggle_lock(self):
        """Toggle lock state"""
        self.locked = not self.locked

        if self.locked:
            self.lock_btn.config(text="🔒 LOCKED", bg='#00ff88')
            self.save_btn.config(state=tk.NORMAL)
            self.main_frame.config(cursor='')
            self.info_label.config(text="✅ LOCKED!\n\nClick SAVE to continue")
        else:
            self.lock_btn.config(text="🔓 UNLOCK - Drag to Position", bg='#ffaa00')
            self.save_btn.config(state=tk.DISABLED)
            self.info_label.config(
                text="⚠ MINIMAP CALIBRATION REQUIRED\n\n"
                     "Drag this over your minimap\n"
                     "Resize from edges to fit\n\n"
                     "Click LOCK → SAVE when done"
            )

    def _update_position_label(self):
        """Update position display"""
        x = self.root.winfo_x()
        y = self.root.winfo_y()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        self.pos_label.config(text=f"Position: ({x}, {y}) | Size: {width}x{height}")
        self.root.after(100, self._update_position_label)

    def _save(self):
        """Save and close"""
        self.result = (
            self.root.winfo_x(),
            self.root.winfo_y(),
            self.root.winfo_width(),
            self.root.winfo_height()
        )
        self.root.quit()
        self.root.destroy()

    def run(self) -> Tuple[int, int, int, int]:
        """
        Show calibration overlay and wait for user to save

        Returns:
            (x, y, width, height) tuple of calibrated region
        """
        self.root.mainloop()
        return self.result or (0, 0, 250, 250)


def calibrate_minimap_region(current_x: int = 0, current_y: int = 0,
                             current_width: int = 250, current_height: int = 250) -> Tuple[int, int, int, int]:
    """
    Run calibration UI and return minimap coordinates

    Args:
        current_x, current_y: Current minimap position
        current_width, current_height: Current minimap size

    Returns:
        (x, y, width, height) tuple
    """
    print("=" * 60)
    print("MINIMAP CALIBRATION")
    print("=" * 60)
    print()
    print("A green overlay window will appear.")
    print("Position it over your League of Legends minimap.")
    print()
    print("Controls:")
    print("  • Drag from center to move")
    print("  • Drag from edges/corners to resize")
    print("  • Click LOCK when positioned")
    print("  • Click SAVE to finish")
    print()
    print("=" * 60)

    overlay = MinimapCalibrationOverlay(current_x, current_y, max(current_width, current_height))
    return overlay.run()
