"""
Dimostrazione interattiva di localizzazione Lidar con rifiuto degli oggetti mobili.

Usa la tastiera per guidare il robot in una mappa conosciuta. Gli oggetti statici
sono punti di riferimento noti. Le persone in movimento sono simulate come cerchi
arancioni. Il robot riceve rilevamenti simili a quelli del Lidar e usa un EKF solo
con i rilevamenti che superano un filtro dinamico basato sulla coerenza del moto relativo.
"""

from __future__ import annotations  # Permette di valutare le annotazioni di tipo in modo differito.

import csv
from dataclasses import dataclass   # Importa il decoratore per definire classi dati di configurazione.
from datetime import datetime
import json
import os                           # Importa gli strumenti del sistema operativo per configurare l'ambiente.
from pathlib import Path            # Importa gli strumenti per costruire percorsi portabili.

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))  # Mantiene i file generati da Matplotlib nella cartella del progetto.

import matplotlib.pyplot as plt                     # Importa pyplot per creare e visualizzare la finestra della simulazione.
from matplotlib.animation import FuncAnimation      # Importa il gestore dell'animazione che richiama ripetutamente update().
import numpy as np                                  # Importa NumPy per vettori, matrici, valori casuali e funzioni trigonometriche.



def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Riporta un angolo nell'intervallo [-pi, pi)."""

    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class ExperimentConfig:
    """Parametri comuni al modello di moto e all'EKF."""

    seed: int = 7
    dt: float = 0.1
    steps: int = 550
    world_size: float = 18.0
    n_landmarks: int = 42
    n_people: int = 6
    lidar_max_range: float = 11.0
    lidar_fov: float = 2.0 * np.pi
    sigma_range: float = 0.12
    sigma_bearing: float = np.deg2rad(1.8)
    process_sigma_xy: float = 0.035
    process_sigma_theta: float = np.deg2rad(1.2)
    base_gate: float = 5.991
    min_gate: float = 3.2
    max_gate: float = 9.5
    gate_speed_gain: float = 0.45
    gate_uncertainty_gain: float = 0.7
    gate_range_tightening: float = 0.025


class PoseEKF:
    """EKF per la localizzazione 2D con misure distanza-angolo."""

    def __init__(self, initial_state: np.ndarray, initial_cov: np.ndarray, cfg: ExperimentConfig):
        self.x = initial_state.astype(float).reshape(3)
        self.P = initial_cov.astype(float)
        self.cfg = cfg

    def predict(self, control: tuple[float, float]) -> None:
        v, omega = control
        x, y, th = self.x
        dt = self.cfg.dt
        self.x = np.array([
            x + v * np.cos(th) * dt,
            y + v * np.sin(th) * dt,
            wrap_angle(th + omega * dt),
        ])
        F = np.array([
            [1.0, 0.0, -v * np.sin(th) * dt],
            [0.0, 1.0, v * np.cos(th) * dt],
            [0.0, 0.0, 1.0],
        ])
        Q = np.diag([
            self.cfg.process_sigma_xy**2,
            self.cfg.process_sigma_xy**2,
            self.cfg.process_sigma_theta**2,
        ])
        self.P = F @ self.P @ F.T + Q

    def measurement_model(self, landmark: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx = landmark[0] - self.x[0]
        dy = landmark[1] - self.x[1]
        q = max(dx * dx + dy * dy, 1e-9)
        r = np.sqrt(q)
        bearing = wrap_angle(np.arctan2(dy, dx) - self.x[2])
        H = np.array([
            [-dx / r, -dy / r, 0.0],
            [dy / q, -dx / q, -1.0],
        ])
        return np.array([r, bearing]), H

    def update(self, z: np.ndarray, landmark: np.ndarray, R: np.ndarray) -> float:
        z_hat, H = self.measurement_model(landmark)
        innovation = z - z_hat
        innovation[1] = wrap_angle(innovation[1])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.x[2] = wrap_angle(self.x[2])
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        return float(innovation.T @ np.linalg.inv(S) @ innovation)

    def innovation_distance(self, z: np.ndarray, landmark: np.ndarray, R: np.ndarray) -> tuple[float, np.ndarray]:
        z_hat, H = self.measurement_model(landmark)
        innovation = z - z_hat
        innovation[1] = wrap_angle(innovation[1])
        S = H @ self.P @ H.T + R
        d2 = float(innovation.T @ np.linalg.inv(S) @ innovation)
        return d2, S


def step_unicycle(state: np.ndarray, control: tuple[float, float], dt: float) -> np.ndarray:
    """Propaga lo stato di un robot con modello cinematico unicycle."""

    v, omega = control
    x, y, th = state
    return np.array([
        x + v * np.cos(th) * dt,
        y + v * np.sin(th) * dt,
        wrap_angle(th + omega * dt),
    ])

PATH = Path(__file__).resolve().parent  # Memorizza la cartella del progetto per salvare file di output.
FOLDER = PATH / "artifacts"             # Crea una sottocartella per salvare i file generati dalla simulazione.


@dataclass
class InteractiveConfig(ExperimentConfig):
    """Valori di configurazione specifici per la visualizzazione interattiva."""

    # Dimensioni della simulazione e della visualizzazione.
    dt: float = 0.08            # Imposta il passo temporale della simulazione in secondi.
    world_size: float = 14.0    # Imposta la semi-larghezza e la semi-altezza del mondo simulato.

    # Numero e parametri di rilevamento degli oggetti statici e mobili.
    n_landmarks: int = 24                       # Memorizza il numero configurato di punti di riferimento statici.
    n_people: int = 4                           # Memorizza il numero di persone mobili da simulare.
    lidar_max_range: float = 8.5                # Limita i rilevamenti simulati a questa distanza in metri.
    sigma_range: float = 0.06                   # Imposta la deviazione standard del rumore sulla distanza in metri.
    sigma_bearing: float = np.deg2rad(1.0)      # Converte un grado di rumore angolare in radianti.

    # Soglie usate per decidere se un rilevamento appartiene a un punto statico.
    relative_motion_gate: float = 0.55  # Imposta la soglia per rifiutare un moto relativo incoerente.
    association_gate: float = 1.1       # Imposta il massimo errore cartesiano per associare un punto di riferimento.


def relative_xy(robot_state: np.ndarray, world_point: np.ndarray) -> np.ndarray:
    """Trasforma un punto del mondo nel sistema di coordinate x-y locale del robot."""

    # Calcola lo spostamento del punto nelle coordinate del mondo.
    dx = world_point[0] - robot_state[0]    # Calcola lo spostamento orizzontale dal robot.
    dy = world_point[1] - robot_state[1]    # Calcola lo spostamento verticale dal robot.

    # Ruota lo spostamento usando l'angolo di orientamento inverso del robot.
    c = np.cos(robot_state[2])  # Calcola il coseno dell'orientamento del robot.
    s = np.sin(robot_state[2])  # Calcola il seno dell'orientamento del robot.

    """[cos -sin] [dx]
       [sin  cos] [dy]   è l'operazione nel return"""
    return np.array([c * dx + s * dy, -s * dx + c * dy])  # Ruota lo spostamento del mondo nelle coordinate del robot.


def relative_to_measurement(rel_xy: np.ndarray, rng: np.random.Generator, cfg: InteractiveConfig) -> np.ndarray:
    """Converte un punto cartesiano locale in una misura rumorosa di distanza e angolo."""

    # Converte le coordinate cartesiane locali in coordinate polari.
    r = np.hypot(rel_xy[0], rel_xy[1])      # Calcola la distanza euclidea del punto dal robot.
    b = np.arctan2(rel_xy[1], rel_xy[0])    # Calcola l'angolo del punto nel sistema del robot.

    # Aggiunge rumore indipendente alla distanza e all'angolo misurati.
    return np.array([                                       # Costruisce il vettore della misura rumorosa distanza-angolo.
        r + rng.normal(0.0, cfg.sigma_range),               # Aggiunge rumore gaussiano alla distanza misurata.
        wrap_angle(b + rng.normal(0.0, cfg.sigma_bearing)), # Aggiunge rumore angolare e normalizza il risultato.
    ])


def measurement_to_relative(z: np.ndarray) -> np.ndarray:
    """Converte una misura distanza-angolo in coordinate cartesiane locali."""

    return np.array([z[0] * np.cos(z[1]), z[0] * np.sin(z[1])])  # Riconverte le coordinate polari in coordinate cartesiane locali.

def save_animation(fig, ani, filename: str) -> None:
    """Salva l'animazione in un file MP4 usando ffmpeg."""

    # Controlla se ffmpeg è disponibile e salva l'animazione.
    try:
        FOLDER.mkdir(parents=True, exist_ok=True)                   # Crea la cartella degli artefatti se non esiste.
        output_path = FOLDER / filename                             # Costruisce il percorso del file nella cartella artifacts.
        ani.save(output_path, writer="ffmpeg", fps=30, dpi=200)     # Salva l'animazione come file MP4.
        print(f"Animazione salvata in {output_path}")               # Conferma il salvataggio del file.
    except Exception as e:
        print(f"Errore durante il salvataggio dell'animazione: {e}")  # Mostra un messaggio di errore se il salvataggio fallisce.

def screenshot_animation(fig, filename: str) -> None:
    """Salva un'istantanea della figura corrente in un file PNG.
       Nella cartella artifacts"""

    try:
        FOLDER.mkdir(parents=True, exist_ok=True)       # Crea la cartella degli artefatti se non esiste.
        output_path = FOLDER / filename                 # Costruisce il percorso del file nella cartella artifacts.
        fig.savefig(output_path, dpi=200)               # Salva la figura corrente come file PNG.
        print(f"Istantanea salvata in {output_path}")   # Conferma il salvataggio del file.
    except Exception as e:
        print(f"Errore durante il salvataggio dell'istantanea: {e}")  # Mostra un messaggio di errore se il salvataggio fallisce.


class InteractiveLidarSimulation:
    """Gestisce lo stato della simulazione, l'EKF e gli elementi grafici Matplotlib."""

    def __init__(self, cfg: InteractiveConfig | None = None):
        """Inizializza il mondo, il robot, il filtro, i contatori e il grafico."""

        # Usa la configurazione fornita oppure i valori predefiniti interattivi.
        self.cfg = cfg or InteractiveConfig()               # Usa le impostazioni ricevute o crea quelle predefinite.
        self.rng = np.random.default_rng(self.cfg.seed)     # Crea un generatore casuale riproducibile usando il seme configurato.

        # Inizializza la posa reale del robot e i comandi controllati dalla tastiera.
        self.robot = np.array([-7.5, -6.0, np.deg2rad(20.0)])   # Memorizza posizione, quota e orientamento reali del robot.
        self.v_cmd = 0.0                                        # Inizia con velocità lineare nulla.
        self.omega_cmd = 0.0                                    # Inizia con velocità angolare nulla.
        self.max_v = 2.0                                        # Limita il comando lineare a due metri al secondo.
        self.max_omega = 1.4                                    # Limita il comando angolare in radianti al secondo.

        # Crea la mappa conosciuta e le persone mobili inizializzate casualmente.
        self.landmarks = self._make_static_objects()    # Costruisce la mappa statica usata per la localizzazione.
        self.people = self.rng.uniform(-0.65 * self.cfg.world_size, 0.65 * self.cfg.world_size, size=(self.cfg.n_people, 2))  # Posiziona casualmente le persone mobili.
        self.people_vel = self.rng.normal(0.0, 0.55, size=(self.cfg.n_people, 2))  # Assegna a ogni persona una velocità casuale bidimensionale.

        # Avvia l'EKF con una posa e una covarianza volutamente imprecise.
        self.ekf = PoseEKF(  # Costruisce il filtro della posa con una stima iniziale spostata.
            initial_state=self.robot + np.array([0.5, -0.35, np.deg2rad(8.0)]), # Sposta la posa iniziale del filtro rispetto a quella reale.
            initial_cov=np.diag([0.7, 0.7, np.deg2rad(10.0)]) ** 2,             # Imposta l'incertezza iniziale indipendente di posizione e orientamento.
            cfg=self.cfg,                                                       # Passa la configurazione interattiva al filtro.
        )

        # Conserva le traiettorie e gli insiemi aggiornati dei rilevamenti visibili e accettati.
        self.true_path = [self.robot.copy()]            # Inizia la storia delle pose reali dalla posa iniziale.
        self.estimated_path = [self.ekf.x.copy()]       # Inizia la storia delle pose stimate dallo stato iniziale del filtro.
        self.visible_static = np.empty((0, 2))          # Riserva un array vuoto per gli oggetti statici visibili.
        self.detected_dynamic = np.empty((0, 2))        # Riserva un array vuoto per i rilevamenti mobili rifiutati.
        self.accepted_static_world = np.empty((0, 2))   # Riserva un array vuoto per le posizioni accettate dei punti di riferimento.
        self.current_lidar_static = 0                   # Inizializza il numero di ritorni statici visibili nel fotogramma corrente.
        self.current_lidar_dynamic = 0                  # Inizializza il numero di ritorni mobili visibili nel fotogramma corrente.
        self.current_gate_static = 0                    # Inizializza il conteggio dei rilevamenti accettati dal filtro.
        self.current_gate_dynamic = 0                   # Inizializza il conteggio dei rilevamenti rifiutati dal filtro.

        self.accepted_count = 0 # Conta tutti i rilevamenti accettati dall'inizio della simulazione.
        self.rejected_count = 0 # Conta tutti i rilevamenti rifiutati dall'inizio della simulazione.
        self.dynamic_count = 0  # Conta i rilevamenti rifiutati che provengono da persone.
        self.frame = 0          # Tiene traccia dei fotogrammi di animazione completati.
        self.error_log: list[dict[str, float | int]] = []
        self._record_error_metrics()

        # Costruisce la figura Matplotlib dopo aver creato tutti i dati da visualizzare.
        self.fig, self.ax = plt.subplots(figsize=(8.8, 8.0))    # Crea la figura e gli assi della visualizzazione.
        self._setup_plot()                                      # Configura gli assi e crea tutti gli elementi grafici aggiornabili.

    def _make_static_objects(self) -> np.ndarray:
        """Costruisce la mappa conosciuta dei punti di riferimento usata dall'EKF."""

        # Mappa conosciuta semplice: file simili a pareti più alcuni oggetti isolati.
        wall_top = np.column_stack((np.linspace(-10, 10, 9), np.full(9, 9.5)))      # Dispone nove punti lungo la parete superiore.
        wall_bottom = np.column_stack((np.linspace(-10, 10, 9), np.full(9, -9.5)))  # Dispone nove punti lungo la parete inferiore.
        columns = np.array([[-8, -2], [-4, 3], [0, -1], [5, 4], [8, -4], [2, 7]], dtype=float)  # Definisce sei punti di riferimento isolati.
        return np.vstack((wall_top, wall_bottom, columns))                          # Unisce tutti i gruppi nella matrice della mappa.

    def _setup_plot(self) -> None:
        """Configura gli assi e crea tutti gli elementi aggiornati dall'animazione."""

        # Imposta i limiti del mondo e la presentazione di base degli assi.
        lim = self.cfg.world_size + 1.0     # Estende leggermente i limiti del mondo per creare margine visivo.
        self.ax.set_xlim(-lim, lim)         # Imposta i limiti dell'asse orizzontale.
        self.ax.set_ylim(-lim, lim)         # Imposta i limiti dell'asse verticale.
        self.ax.set_aspect("equal")         # Usa la stessa scala per mostrare correttamente le distanze.
        self.ax.grid(True, alpha=0.25)      # Disegna una griglia tenue per facilitare la lettura delle posizioni.
        self.ax.set_title("EKF con Gate Dinamico: guida il robot con la tastiera")  # Aggiunge il titolo al grafico.
        self.ax.set_xlabel("x [m]")         # Etichetta l'asse orizzontale in metri.
        self.ax.set_ylabel("y [m]")         # Etichetta l'asse verticale in metri.

        # Crea gli elementi grafici per punti di riferimento, rilevamenti, persone e pose del robot.
        self.map_scatter = self.ax.scatter(self.landmarks[:, 0], self.landmarks[:, 1], marker="s", s=60, c="#59656f", label="oggetti statici noti")     # Disegna tutti i punti di riferimento conosciuti.
        self.visible_scatter = self.ax.scatter([], [], marker="s", s=90, facecolors="none", edgecolors="#009e73", linewidths=2, label="statici usati")  # Prepara l'insieme dei punti accettati.
        self.people_scatter = self.ax.scatter(self.people[:, 0], self.people[:, 1], marker="o", s=85, c="#d55e00", label="persone")                     # Disegna le persone mobili.
        self.dynamic_scatter = self.ax.scatter([], [], marker="x", s=90, c="#cc3311", linewidths=2, label="dinamici scartati")                          # Prepara i marcatori dei rilevamenti rifiutati.

        self.robot_dot, = self.ax.plot([], [], "o", color="#0072b2", markersize=9, label="robot vero")  # Crea il marcatore della posizione reale.
        self.robot_head, = self.ax.plot([], [], "-", color="#0072b2", linewidth=2)                      # Crea la linea dell'orientamento reale.
        self.ekf_dot, = self.ax.plot([], [], "o", color="#cc79a7", markersize=7, label="posa EKF")      # Crea il marcatore della stima EKF.
        self.ekf_head, = self.ax.plot([], [], "-", color="#cc79a7", linewidth=2)                        # Crea la linea dell'orientamento stimato.
        self.true_line, = self.ax.plot([], [], "-", color="#111111", linewidth=1.3, alpha=0.75)         # Crea la traiettoria reale.
        self.ekf_line, = self.ax.plot([], [], "--", color="#cc79a7", linewidth=1.5)                     # Crea la traiettoria stimata.

        # Mostra il raggio di rilevamento corrente attorno alla posa reale del robot.
        self.lidar_circle = plt.Circle((0, 0), self.cfg.lidar_max_range, fill=False, color="#0072b2", alpha=0.18)  # Rappresenta la portata massima del sensore.
        self.ax.add_patch(self.lidar_circle)  # Aggiunge il cerchio della portata agli assi.

        self.info = self.ax.text(           # Crea l'area di testo per le statistiche aggiornate della simulazione.
            0.02,                           # Posiziona il testo vicino al bordo sinistro degli assi.
            0.98,                           # Posiziona il testo vicino al bordo superiore degli assi.
            "",                             # Inizia con testo vuoto fino al disegno del primo fotogramma.
            transform=self.ax.transAxes,    # Interpreta la posizione in coordinate relative agli assi.
            va="top",                       # Ancora il bordo superiore del testo alla coordinata verticale.
            fontsize=9,                     # Usa una dimensione compatta e leggibile per le informazioni.
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),  # Fornisce uno sfondo traslucido leggibile.
        )

        # Aggiunge la legenda e inoltra gli eventi della tastiera al controllore.
        self.ax.legend(loc="lower right", fontsize=8)               # Mostra le etichette di tutti i tipi di oggetti disegnati.
        self.fig.canvas.mpl_connect("key_press_event", self.on_key) # Inoltra gli eventi della tastiera al gestore dei comandi.

    def on_key(self, event) -> None:
        """Aggiorna i comandi di velocità o resetta l'EKF in risposta alla tastiera."""

        # I tasti freccia e WASD regolano le velocità lineare e angolare richieste.
        if event.key in ("up", "w"):
            self.v_cmd = min(self.max_v, self.v_cmd + 0.25) # Aumenta la velocità in avanti senza superare il limite.
        elif event.key in ("down", "s"):
            self.v_cmd = max(-0.8, self.v_cmd - 0.25)       # Riduce la velocità senza superare il limite in retromarcia.
        elif event.key in ("left", "a"):
            self.omega_cmd = min(self.max_omega, self.omega_cmd + 0.18)     # Aumenta la velocità di rotazione a sinistra entro il limite.
        elif event.key in ("right", "d"):
            self.omega_cmd = max(-self.max_omega, self.omega_cmd - 0.18)    # Aumenta la velocità di rotazione a destra entro il limite.
        elif event.key == " ":
            # La barra spaziatrice ordina l'arresto immediato.
            self.v_cmd = 0.0        # Arresta immediatamente il movimento in avanti.
            self.omega_cmd = 0.0    # Arresta immediatamente la rotazione.
        elif event.key == "r":
            # Resetta solo la stima del filtro; il robot simulato continua a muoversi.
            self.ekf.x = self.robot + np.array([0.25, -0.2, np.deg2rad(4.0)])   # Riporta la stima vicino alla posa reale corrente.
            self.ekf.P = np.diag([0.7, 0.7, np.deg2rad(10.0)]) ** 2             # Ripristina l'incertezza iniziale del filtro.
            self.estimated_path = [self.ekf.x.copy()]                           # Inizia una nuova traiettoria stimata dallo stato resettato.
        elif event.key == "v":
            save_animation(self.fig, self.ani, "lidar_dynamic_gate_video.mp4")
        elif event.key == "escape":
            # Escape chiude la finestra e termina la simulazione.
            screenshot_animation(self.fig, "lidar_dynamic_gate_screenshot.png") # Salva un'istantanea della figura corrente prima di chiudere.
            plt.close(self.fig)                                                 # Chiude la finestra interattiva quando viene premuto Escape.

    def _move_people(self) -> None:
        """Fa avanzare le persone mobili e le riflette ai confini del mondo."""

        # Integra la velocità costante di ogni persona per un passo di simulazione.
        self.people += self.people_vel * self.cfg.dt    # Aggiorna la posizione di ogni persona per un passo temporale.
        limit = 0.8 * self.cfg.world_size               # Mantiene le persone dentro un confine interno leggermente più piccolo.

        # Inverte la velocità quando una persona raggiunge un confine.
        for i in range(len(self.people)):   # Visita ogni persona mobile.
            for axis in range(2):           # Controlla separatamente le coordinate orizzontale e verticale.
                if self.people[i, axis] < -limit or self.people[i, axis] > limit:       # Rileva il superamento di un confine.
                    self.people_vel[i, axis] *= -1.0                                    # Inverte la velocità lungo l'asse attraversato.
                    self.people[i, axis] = np.clip(self.people[i, axis], -limit, limit) # Riporta la persona dentro il confine.

    def _measure_points(self, points: np.ndarray, source: str) -> list[dict]:
        """Genera misure rumorose per i punti entro la portata del Lidar."""

        measurements = []                       # Raccoglie tutti i rilevamenti rumorosi entro la portata per questa sorgente.
        for idx, point in enumerate(points):    # Elabora ogni posizione di punto di riferimento o persona.
            # Trasforma il punto nel sistema del robot e applica il limite di distanza.
            rel = relative_xy(self.robot, point)    # Esprime il punto del mondo nel sistema locale del robot.
            dist = np.linalg.norm(rel)              # Calcola la distanza del punto dal robot.
            if dist <= self.cfg.lidar_max_range:    # Conserva solo i punti visibili entro la portata del sensore.
                # Conserva la sorgente e il punto originale per la classificazione successiva.
                z = relative_to_measurement(rel, self.rng, self.cfg)  # Converte il punto locale in una misura rumorosa.
                measurements.append({"z": z, "source": source, "index": idx, "point": point.copy()})  # Memorizza i dati della misura per la classificazione.
        return measurements

    def _classify_and_update(self, measurements: list[dict], noisy_control: tuple[float, float]) -> None:
        """Associa i rilevamenti ai punti di riferimento e li aggiorna o li rifiuta."""

        # La covarianza di misura dell'EKF è condivisa da tutti i rilevamenti.
        R = np.diag([self.cfg.sigma_range**2, self.cfg.sigma_bearing**2])  # Costruisce la covarianza del rumore distanza-angolo.
        accepted_static = []    # Raccoglie i punti che superano entrambi i filtri.
        dynamic_world = []      # Raccoglie i rilevamenti rifiutati proiettati nelle coordinate del mondo.

        # Verifica ogni misura rispetto alla stessa posa prevista prima dell'aggiornamento.
        predicted_pose = self.ekf.x.copy()              # Fissa la posa usata per valutare le associazioni in modo coerente.
        for measurement in measurements:                # Classifica ogni ritorno simulato del sensore.
            z = measurement["z"]                        # Estrae l'osservazione distanza-angolo.
            observed_rel = measurement_to_relative(z)   # Converte l'osservazione in coordinate cartesiane locali.

            # Valuta il rilevamento rispetto a ogni punto di riferimento conosciuto.
            candidates = []                                 # Memorizza i punteggi di associazione per ogni punto possibile.
            for i, landmark in enumerate(self.landmarks):   # Confronta il rilevamento con ogni punto noto.
                expected_rel = relative_xy(predicted_pose, landmark)        # Prevede la posizione locale di ogni punto.
                rel_error = np.linalg.norm(observed_rel - expected_rel)     # Misura la differenza cartesiana diretta.
                d2, S = self.ekf.innovation_distance(z, landmark, R)        # Calcola la distanza d'innovazione e la relativa covarianza.
                gate = self._dynamic_motion_gate(S, z[0], abs(noisy_control[0]))    # Adatta la soglia a incertezza, velocità e distanza.
                candidates.append((rel_error, d2, gate, i))                         # Salva i valori necessari per scegliere e verificare il candidato.

            # Sceglie il candidato geometricamente più vicino e applica entrambi i filtri.
            rel_error, d2, gate, best_idx = min(candidates, key=lambda item: item[0])   # Seleziona il punto geometricamente più vicino.
            is_static_consistent = rel_error < self.cfg.association_gate and d2 < gate  # Richiede coerenza geometrica e statistica.

            if is_static_consistent:
                # Un rilevamento coerente viene trattato come ritorno di un punto statico.
                self.ekf.update(z, self.landmarks[best_idx], R)     # Integra l'osservazione affidabile nell'EKF.
                accepted_static.append(self.landmarks[best_idx])    # Registra il punto accettato per la visualizzazione.
                self.accepted_count += 1                            # Incrementa il contatore totale delle misure accettate.
            else:
                # I rilevamenti rifiutati vengono mostrati nelle coordinate del mondo.
                dynamic_world.append(self._measurement_world_point(z))  # Converte e registra il ritorno rifiutato per visualizzarlo.
                self.rejected_count += 1                                # Incrementa il contatore totale delle misure rifiutate.
                if measurement["source"] == "person":                   # Verifica se il ritorno rifiutato proviene da una persona simulata.
                    self.dynamic_count += 1                             # Incrementa il contatore delle persone rilevate.

        self.accepted_static_world = np.array(accepted_static) if accepted_static else np.empty((0, 2)) # Memorizza le coordinate accettate oppure un array vuoto.
        self.detected_dynamic = np.array(dynamic_world) if dynamic_world else np.empty((0, 2))          # Memorizza le coordinate rifiutate oppure un array vuoto.
        self.current_gate_static = len(accepted_static)                                                 # Riporta i rilevamenti accettati nel fotogramma corrente.
        self.current_gate_dynamic = len(dynamic_world)                                                  # Riporta i rilevamenti rifiutati nel fotogramma corrente.

    def _dynamic_motion_gate(self, S: np.ndarray, measured_range: float, speed: float) -> float:
        """Calcola una soglia d'innovazione limitata in base a incertezza e movimento."""

        # Una maggiore incertezza o velocità amplia la soglia; la distanza la restringe.
        uncertainty = np.sqrt(max(np.linalg.det(S), 0.0))                       # Converte il volume di covarianza in una scala di incertezza non negativa.
        threshold = (self.cfg.base_gate
                    + self.cfg.gate_uncertainty_gain * uncertainty
                    + self.cfg.gate_speed_gain * speed
                    - self.cfg.gate_range_tightening * measured_range
                    )
                   # Combina incertezza, velocità e distanza nella soglia.
        return float(np.clip(threshold, self.cfg.min_gate, self.cfg.max_gate))  # Limita la soglia all'intervallo configurato.

    def _measurement_world_point(self, z: np.ndarray) -> np.ndarray:
        """Proietta un rilevamento locale nelle coordinate del mondo per visualizzarlo."""

        # Riconverte la misura in un vettore cartesiano locale.
        rel = measurement_to_relative(z)    # Converte la misura locale distanza-angolo in un vettore cartesiano.

        # Ruota il vettore usando l'orientamento del robot e lo trasla nel mondo.
        c = np.cos(self.robot[2])           # Calcola il coseno dell'orientamento del robot nel mondo.
        s = np.sin(self.robot[2])           # Calcola il seno dell'orientamento del robot nel mondo.
        return self.robot[:2] + np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])  # Ruota nelle coordinate del mondo e trasla dal robot.

    def _record_error_metrics(self) -> None:
        """Registra errori di posa e conteggi per il fotogramma corrente."""

        heading_error = float(wrap_angle(self.ekf.x[2] - self.robot[2]))
        position_error = self.ekf.x[:2] - self.robot[:2]
        self.error_log.append({
            "frame": self.frame,
            "time_s": self.frame * self.cfg.dt,
            "true_x_m": float(self.robot[0]),
            "true_y_m": float(self.robot[1]),
            "true_heading_rad": float(self.robot[2]),
            "estimated_x_m": float(self.ekf.x[0]),
            "estimated_y_m": float(self.ekf.x[1]),
            "estimated_heading_rad": float(self.ekf.x[2]),
            "error_x_m": float(position_error[0]),
            "error_y_m": float(position_error[1]),
            "position_error_m": float(np.linalg.norm(position_error)),
            "heading_error_rad": heading_error,
            "heading_error_deg": float(np.rad2deg(heading_error)),
            "visible_static_count": self.current_lidar_static,
            "visible_dynamic_count": self.current_lidar_dynamic,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "rejected_person_count": self.dynamic_count,
        })

    def _save_error_log(self) -> tuple[Path, Path]:
        """Salva i dati per fotogramma in CSV e le metriche aggregate in JSON."""

        FOLDER.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        csv_path = FOLDER / f"lidar_dynamic_gate_log_{timestamp}.csv"
        summary_path = FOLDER / f"lidar_dynamic_gate_summary_{timestamp}.json"

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.error_log[0].keys())
            writer.writeheader()
            writer.writerows(self.error_log)

        position_errors = np.array([row["position_error_m"] for row in self.error_log])
        heading_errors_deg = np.array([row["heading_error_deg"] for row in self.error_log])
        summary = {
            "seed": self.cfg.seed,
            "dt_s": self.cfg.dt,
            "samples_including_initial_pose": len(self.error_log),
            "duration_s": self.error_log[-1]["time_s"],
            "position_rmse_m": float(np.sqrt(np.mean(position_errors**2))),
            "position_mean_error_m": float(np.mean(position_errors)),
            "position_max_error_m": float(np.max(position_errors)),
            "position_final_error_m": float(position_errors[-1]),
            "heading_rmse_deg": float(np.sqrt(np.mean(heading_errors_deg**2))),
            "heading_mean_absolute_error_deg": float(np.mean(np.abs(heading_errors_deg))),
            "heading_max_absolute_error_deg": float(np.max(np.abs(heading_errors_deg))),
            "accepted_measurements": self.accepted_count,
            "rejected_measurements": self.rejected_count,
            "rejected_person_measurements": self.dynamic_count,
        }
        with summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2)
            summary_file.write("\n")

        print("\nMetriche della simulazione:")
        print(f"  RMSE posizione: {summary['position_rmse_m']:.3f} m")
        print(f"  Errore posizione medio: {summary['position_mean_error_m']:.3f} m")
        print(f"  Errore posizione massimo: {summary['position_max_error_m']:.3f} m")
        print(f"  Errore posizione finale: {summary['position_final_error_m']:.3f} m")
        print(f"  RMSE orientamento: {summary['heading_rmse_deg']:.3f} deg")
        print(f"  Log CSV: {csv_path.resolve()}")
        print(f"  Riepilogo JSON: {summary_path.resolve()}")
        return csv_path, summary_path

    def update(self, _frame):
        """Avanza di un fotogramma, elabora le misure e ridisegna la scena."""

        # Muove il robot reale usando esattamente il comando della tastiera.
        true_control = (self.v_cmd, self.omega_cmd)                         # Raggruppa i comandi per il modello di moto reale.
        self.robot = step_unicycle(self.robot, true_control, self.cfg.dt)   # Fa avanzare il robot simulato di un passo temporale.
        self.robot[0:2] = np.clip(self.robot[0:2], -self.cfg.world_size, self.cfg.world_size)  # Mantiene la posizione del robot entro i limiti del mondo.

        # Fornisce all'EKF comandi leggermente rumorosi per simulare errori di attuazione.
        noisy_control = (                                           # Crea il comando imperfetto fornito all'EKF.
            self.v_cmd + self.rng.normal(0.0, 0.025),               # Aggiunge rumore al comando lineare.
            self.omega_cmd + self.rng.normal(0.0, np.deg2rad(0.5))  # Aggiunge rumore angolare in radianti al secondo.
        )
        self.ekf.predict(noisy_control)                             # Propaga posa e covarianza stimate usando il comando rumoroso.

        # Muove gli oggetti dinamici prima di acquisire la misura del sensore.
        self._move_people()  # Fa avanzare gli oggetti mobili prima di generare le misure del fotogramma.

        # Unisce i ritorni statici e dinamici, poi casualizza l'ordine di elaborazione.
        static_measurements = self._measure_points(self.landmarks, "landmark")  # Simula i ritorni dei punti statici conosciuti.
        dynamic_measurements = self._measure_points(self.people, "person")      # Simula i ritorni delle persone mobili.
        self.current_lidar_static = len(static_measurements)                    # Conta i ritorni statici visibili.
        self.current_lidar_dynamic = len(dynamic_measurements)                  # Conta i ritorni mobili visibili.
        measurements = static_measurements + dynamic_measurements               # Unisce le due sorgenti in un unico flusso di misure.
        self.rng.shuffle(measurements)                                          # Mescola l'ordine per evitare dipendenze dalla sorgente.
        self._classify_and_update(measurements, noisy_control)                  # Associa, filtra ed elabora ogni rilevamento corrente.

        # Salva le traiettorie, ridisegna gli elementi e aggiorna il contatore dei fotogrammi.
        self.true_path.append(self.robot.copy())        # Salva la posa reale aggiornata per la traiettoria.
        self.estimated_path.append(self.ekf.x.copy())   # Salva la posa stimata aggiornata per la traiettoria.
        self.frame += 1                                 # Incrementa il contatore dei fotogrammi completati.
        self._record_error_metrics()                    # Conserva la posa e gli errori per il report.
        self._draw()                                    # Aggiorna tutti gli elementi dinamici e il testo di stato.
        return (                                        # Restituisce tutti gli elementi che rappresentano lo stato aggiornato.
            self.robot_dot,                             # Restituisce il marcatore del robot reale.
            self.robot_head,                            # Restituisce la linea dell'orientamento reale.
            self.ekf_dot,                               # Restituisce il marcatore dell'EKF.
            self.ekf_head,                              # Restituisce la linea dell'orientamento stimato.
            self.true_line,                             # Restituisce la traiettoria reale.
            self.ekf_line,                              # Restituisce la traiettoria stimata.
            self.people_scatter,                        # Restituisce l'insieme delle persone mobili.
            self.visible_scatter,                       # Restituisce l'insieme dei punti accettati.
            self.dynamic_scatter,                       # Restituisce l'insieme dei rilevamenti rifiutati.
            self.info,                                  # Restituisce il testo di stato aggiornato.
            self.lidar_circle,                          # Restituisce il cerchio della portata del sensore.
        )

    def _draw_pose(self, pose: np.ndarray, dot, head, length: float) -> None:
        """Aggiorna un marcatore di posa e la relativa linea di orientamento."""

        dot.set_data([pose[0]], [pose[1]])  # Posiziona il marcatore alle coordinate x-y dello stato.
        head.set_data([pose[0], pose[0] + length * np.cos(pose[2])], [pose[1], pose[1] + length * np.sin(pose[2])]) # Disegna la linea che mostra l'orientamento.

    def _draw(self) -> None:
        """Aggiorna tutti gli elementi dinamici del grafico e il testo di stato."""

        # Aggiorna gli insiemi di punti e i due indicatori della posa del robot.
        self.people_scatter.set_offsets(self.people)                        # Sposta i marcatori delle persone nelle posizioni aggiornate.
        self.visible_scatter.set_offsets(self.accepted_static_world)        # Mostra i punti accettati dal filtro.
        self.dynamic_scatter.set_offsets(self.detected_dynamic)             # Mostra i rilevamenti rifiutati come dinamici o incoerenti.
        self._draw_pose(self.robot, self.robot_dot, self.robot_head, 0.75)  # Disegna la posa reale del robot.
        self._draw_pose(self.ekf.x, self.ekf_dot, self.ekf_head, 0.65)      # Disegna la stima della posa dell'EKF.
        self.lidar_circle.center = tuple(self.robot[:2])                    # Sposta il cerchio della portata insieme al robot reale.

        # Converte le traiettorie memorizzate in array per aggiornare le linee Matplotlib.
        true_path = np.array(self.true_path)                        # Converte la storia delle pose reali in un array indicizzabile per colonne.
        ekf_path = np.array(self.estimated_path)                    # Converte la storia delle pose stimate in un array indicizzabile per colonne.
        self.true_line.set_data(true_path[:, 0], true_path[:, 1])   # Aggiorna le coordinate della traiettoria reale.
        self.ekf_line.set_data(ekf_path[:, 0], ekf_path[:, 1])      # Aggiorna le coordinate della traiettoria stimata.

        # Mostra l'errore di localizzazione corrente e i contatori delle classificazioni.
        err = np.linalg.norm(self.robot[:2] - self.ekf.x[:2])   # Calcola l'errore corrente di localizzazione sul piano.
        self.info.set_text(                                     # Mostra comandi, errore e conteggi correnti e totali dei rilevamenti.
            "Frecce/WASD: guida | Spazio: stop | R: reset EKF\n"
            f"v={self.v_cmd:+.2f} m/s, omega={np.rad2deg(self.omega_cmd):+.1f} deg/s\n"
            f"errore posa={err:.2f} m\n"
            f"Lidar ora: statici={self.current_lidar_static}, dinamici={self.current_lidar_dynamic}\n"
            f"Gate ora: usati={self.current_gate_static}, scartati={self.current_gate_dynamic}\n"
            f"statici accettati={self.accepted_count}, misure scartate={self.rejected_count}, persone rilevate={self.dynamic_count}"
        )

    def run(self) -> None:
        """Avvia l'animazione Matplotlib e attende fino alla chiusura della finestra."""

        # Permette a Matplotlib di chiamare update all'intervallo configurato.
        self.ani = FuncAnimation(self.fig, self.update, interval=1000 * self.cfg.dt, blit=False, cache_frame_data=False)    # Pianifica update() all'intervallo della simulazione.
        plt.show()  # Apre la finestra e attende che l'utente la chiuda.
        self._save_error_log()


def main() -> None:
    """Crea ed esegue la simulazione interattiva con le impostazioni predefinite."""

    InteractiveLidarSimulation().run()  # Crea la simulazione predefinita e avvia l'animazione.


if __name__ == "__main__":
    main()  # Esegue il programma solo quando questo file viene avviato direttamente.
