from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict

from PySide6.QtCore import QTimer, Qt, QEvent
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


STATE_FILE_NAME = "tamagotchi_state.json"


def clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


@dataclass
class Pet:
    name: str = "Tama"
    hunger: float = 30.0
    happiness: float = 70.0
    energy: float = 70.0
    cleanliness: float = 70.0
    age_minutes: float = 0.0
    alive: bool = True
    sleeping: bool = False

    def to_serializable(self) -> Dict[str, float | str | bool]:
        return asdict(self)

    @staticmethod
    def from_serializable(data: Dict[str, float | str | bool]) -> "Pet":
        pet = Pet()
        for key, value in data.items():
            if hasattr(pet, key):
                setattr(pet, key, value)
        return pet

    def apply_natural_decay(self) -> None:
        if not self.alive:
            return

        if self.sleeping:
            # Restorative while sleeping
            self.energy = clamp(self.energy + 3.0)
            self.hunger = clamp(self.hunger + 1.0)
            self.happiness = clamp(self.happiness + 0.2)
            self.cleanliness = clamp(self.cleanliness - 0.2)
        else:
            # Awake decay
            self.hunger = clamp(self.hunger + 0.5)
            self.energy = clamp(self.energy - 0.4)
            self.happiness = clamp(self.happiness - 0.2)
            self.cleanliness = clamp(self.cleanliness - 0.2)

            if self.hunger > 80:
                self.happiness = clamp(self.happiness - 0.6)
            if self.energy < 30:
                self.happiness = clamp(self.happiness - 0.4)
            if self.cleanliness < 30:
                self.happiness = clamp(self.happiness - 0.4)

        self.age_minutes = max(0.0, self.age_minutes + 1.0 / 60.0)
        self._update_alive_status()

    def feed(self) -> None:
        if not self.alive:
            return
        self.hunger = clamp(self.hunger - 25.0)
        self.cleanliness = clamp(self.cleanliness - 5.0)
        self.happiness = clamp(self.happiness + 5.0)
        self._update_alive_status()

    def play(self) -> None:
        if not self.alive:
            return
        if self.sleeping:
            return
        self.happiness = clamp(self.happiness + 20.0)
        self.energy = clamp(self.energy - 15.0)
        self.hunger = clamp(self.hunger + 10.0)
        self.cleanliness = clamp(self.cleanliness - 5.0)
        self._update_alive_status()

    def clean_up(self) -> None:
        if not self.alive:
            return
        self.cleanliness = clamp(self.cleanliness + 30.0)
        self.happiness = clamp(self.happiness + 5.0)
        self._update_alive_status()

    def start_sleep(self) -> None:
        if not self.alive:
            return
        if self.sleeping:
            return
        self.sleeping = True

    def stop_sleep(self) -> None:
        if not self.alive:
            return
        if not self.sleeping:
            return
        self.sleeping = False

    def reset(self) -> None:
        self.hunger = 30.0
        self.happiness = 70.0
        self.energy = 70.0
        self.cleanliness = 70.0
        self.age_minutes = 0.0
        self.alive = True
        self.sleeping = False

    def health_score(self) -> float:
        return clamp((100.0 - self.hunger) * 0.35 + self.happiness * 0.25 + self.energy * 0.25 + self.cleanliness * 0.15)

    def mood_emoji(self) -> str:
        if not self.alive:
            return "💀"
        if self.sleeping:
            return "😴"
        if self.hunger > 85:
            return "😣"
        if self.cleanliness < 25:
            return "🤢"
        if self.energy < 25:
            return "🥱"
        if self.happiness > 75 and self.health_score() > 70:
            return "😄"
        if self.happiness < 35:
            return "😢"
        return "🙂"

    def _update_alive_status(self) -> None:
        if any(
            math.isclose(value, 0.0) or value <= 0.0
            for value in [100.0 - self.hunger, self.happiness, self.energy, self.cleanliness]
        ):
            self.alive = False
            self.sleeping = False


class GameWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Tamagotchi 🐣")
        self.setMinimumSize(520, 420)

        self.pet: Pet = self._load_state()

        self._build_ui()
        self._connect_signals()
        self._update_view()

        self.tick_timer = QTimer(self)
        self.tick_timer.setInterval(1000)
        self.tick_timer.timeout.connect(self._on_tick)
        self.tick_timer.start()

        self.sleep_timer = QTimer(self)
        self.sleep_timer.setInterval(10000)
        self.sleep_timer.setSingleShot(True)
        self.sleep_timer.timeout.connect(self._finish_sleep)

        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(5000)
        self.autosave_timer.timeout.connect(self._save_state)
        self.autosave_timer.start()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        self.pet_label = QLabel(self)
        self.pet_label.setAlignment(Qt.AlignCenter)
        self.pet_label.setStyleSheet(
            "font-size: 64px; line-height: 64px; margin: 6px 0;"
        )

        self.status_label = QLabel(self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; color: #444;")

        root_layout.addWidget(self.pet_label)
        root_layout.addWidget(self.status_label)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        self.hunger_bar = self._create_bar()
        self.happiness_bar = self._create_bar()
        self.energy_bar = self._create_bar()
        self.cleanliness_bar = self._create_bar()

        grid.addWidget(self._create_label("Hunger"), 0, 0)
        grid.addWidget(self.hunger_bar, 0, 1)

        grid.addWidget(self._create_label("Happiness"), 1, 0)
        grid.addWidget(self.happiness_bar, 1, 1)

        grid.addWidget(self._create_label("Energy"), 2, 0)
        grid.addWidget(self.energy_bar, 2, 1)

        grid.addWidget(self._create_label("Cleanliness"), 3, 0)
        grid.addWidget(self.cleanliness_bar, 3, 1)

        root_layout.addLayout(grid)

        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(8)

        self.feed_button = QPushButton("🍙 Feed")
        self.play_button = QPushButton("🎾 Play")
        self.clean_button = QPushButton("🫧 Clean")
        self.sleep_button = QPushButton("🛌 Sleep")
        self.wake_button = QPushButton("⏰ Wake")
        self.reset_button = QPushButton("🔁 Reset")

        self.wake_button.setEnabled(False)

        for widget in [
            self.feed_button,
            self.play_button,
            self.clean_button,
            self.sleep_button,
            self.wake_button,
            self.reset_button,
        ]:
            actions_layout.addWidget(widget)

        root_layout.addLayout(actions_layout)

        self._install_menu()

        self.setStyleSheet(
            """
            QProgressBar { height: 16px; border: 1px solid #bbb; border-radius: 4px; background: #f3f3f3; }
            QProgressBar::chunk { background-color: #8ecae6; }
            QPushButton { padding: 8px 10px; }
            """
        )

    def _install_menu(self) -> None:
        menu = self.menuBar().addMenu("Game")
        save_action = QAction("Save", self)
        save_action.triggered.connect(self._save_state)
        menu.addAction(save_action)

        load_action = QAction("Load", self)
        load_action.triggered.connect(self._reload_state)
        menu.addAction(load_action)

        menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        menu.addAction(exit_action)

    def _create_bar(self) -> QProgressBar:
        bar = QProgressBar(self)
        bar.setRange(0, 100)
        return bar

    def _create_label(self, text: str) -> QLabel:
        label = QLabel(text, self)
        label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        label.setStyleSheet("font-weight: 600; margin-right: 8px;")
        return label

    def _connect_signals(self) -> None:
        self.feed_button.clicked.connect(self._on_feed)
        self.play_button.clicked.connect(self._on_play)
        self.clean_button.clicked.connect(self._on_clean)
        self.sleep_button.clicked.connect(self._on_sleep)
        self.wake_button.clicked.connect(self._on_wake)
        self.reset_button.clicked.connect(self._on_reset)

    def _on_tick(self) -> None:
        self.pet.apply_natural_decay()
        self._update_view()
        if not self.pet.alive:
            self._handle_game_over()

    def _on_feed(self) -> None:
        self.pet.feed()
        self._update_view()

    def _on_play(self) -> None:
        self.pet.play()
        self._update_view()

    def _on_clean(self) -> None:
        self.pet.clean_up()
        self._update_view()

    def _on_sleep(self) -> None:
        if not self.pet.sleeping and self.pet.alive:
            self.pet.start_sleep()
            self.sleep_timer.start()
            self._update_view()

    def _finish_sleep(self) -> None:
        self.pet.stop_sleep()
        self._update_view()

    def _on_wake(self) -> None:
        self.sleep_timer.stop()
        self.pet.stop_sleep()
        self._update_view()

    def _on_reset(self) -> None:
        self.pet.reset()
        self._update_view()

    def _update_view(self) -> None:
        self.pet_label.setText(self.pet.mood_emoji())

        # Hunger bar shows "how full" rather than "how hungry"
        self.hunger_bar.setValue(int(round(100.0 - self.pet.hunger)))
        self.happiness_bar.setValue(int(round(self.pet.happiness)))
        self.energy_bar.setValue(int(round(self.pet.energy)))
        self.cleanliness_bar.setValue(int(round(self.pet.cleanliness)))

        age_text = f"Age: {self.pet.age_minutes:.1f} min"
        health_text = f"Health: {self.pet.health_score():.0f}/100"
        state_text = "Sleeping" if self.pet.sleeping else "Awake"
        self.status_label.setText(f"{self.pet.name} — {state_text}  |  {age_text}  |  {health_text}")

        is_alive = self.pet.alive
        is_sleeping = self.pet.sleeping

        self.feed_button.setEnabled(is_alive and not is_sleeping)
        self.play_button.setEnabled(is_alive and not is_sleeping)
        self.clean_button.setEnabled(is_alive and not is_sleeping)
        self.sleep_button.setEnabled(is_alive and not is_sleeping)
        self.wake_button.setEnabled(is_alive and is_sleeping)
        self.reset_button.setEnabled(True)

    def _handle_game_over(self) -> None:
        self.tick_timer.stop()
        self.sleep_timer.stop()
        self.autosave_timer.stop()
        self._save_state()
        QMessageBox.information(self, "Game Over", f"{self.pet.name} has passed away... 💔\nPress Reset to start again.")
        self._update_view()

    def _save_state(self) -> None:
        try:
            data = self.pet.to_serializable()
            with open(self._state_file_path(), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            # Silent save failure; avoid interrupting gameplay
            pass

    def _reload_state(self) -> None:
        self.pet = self._load_state()
        self._update_view()

    def _load_state(self) -> Pet:
        try:
            path = self._state_file_path()
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                pet = Pet.from_serializable(data)
                if not isinstance(pet.name, str) or not pet.name:
                    pet.name = "Tama"
                return pet
        except Exception:
            pass
        return Pet()

    def _state_file_path(self) -> str:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, STATE_FILE_NAME)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt API)
        self._save_state()
        super().closeEvent(event)


def main() -> None:
    app = QApplication.instance() or QApplication([])
    window = GameWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()


