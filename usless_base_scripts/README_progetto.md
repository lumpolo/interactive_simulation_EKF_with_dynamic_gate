# Localizzazione in Ambiente Dinamico: Rilevamento di Persone

Obiettivo: usare misure Lidar per localizzare un robot in una mappa nota, rilevando ed escludendo oggetti mobili, per esempio persone, dalle correzioni dell'EKF. La mappa contiene oggetti statici noti, simulati come muri, cubi o sfere. Il robot osserva la posizione relativa degli oggetti; se un oggetto osservato non si muove come dovrebbe muoversi un oggetto statico rispetto al robot, viene classificato come dinamico e scartato.

## Idea del metodo

Lo stato stimato nell'esperimento Python e':

```text
x = [px, py, theta]
```

Il modello di predizione usa cinematica unicycle e odometria rumorosa:

```text
px(k+1)    = px(k) + v cos(theta) dt
py(k+1)    = py(k) + v sin(theta) dt
theta(k+1) = theta(k) + omega dt
```

Il Lidar genera misure range-bearing verso oggetti statici della mappa. In piu', alcune persone mobili, simulate come sfere o cubi, generano misure simili, ma non hanno una corrispondenza coerente nella mappa statica. Se una misura mobile venisse usata come oggetto statico, l'EKF correggerebbe la posa nella direzione sbagliata.

Per ogni misura si calcola la migliore associazione con i landmark della mappa tramite distanza di Mahalanobis:

```text
d^2 = innovazione^T S^-1 innovazione
S = H P H^T + R
```

La misura viene usata solo se `d^2` e' sotto la soglia del gate e se la posizione relativa osservata e' coerente con quella prevista per un oggetto statico.

## Criterio del professore: moto relativo degli oggetti

Se il robot si muove da una posa predetta `x(k)` a `x(k+1)`, un oggetto statico noto deve cambiare posizione relativa solo a causa del moto del robot. Quindi, per ogni oggetto della mappa si puo' predire dove dovrebbe apparire nel frame del robot:

```text
p_rel_pred = R(theta_robot)^T (p_oggetto_statico - p_robot_pred)
```

La misura Lidar fornisce invece:

```text
p_rel_obs = [range cos(bearing), range sin(bearing)]
```

Se la distanza tra `p_rel_obs` e `p_rel_pred` e' piccola, l'oggetto e' compatibile con la mappa statica e puo' essere usato dall'EKF. Se la distanza e' grande, l'oggetto si e' mosso piu' o meno di quanto previsto dal solo moto del robot: viene trattato come persona/oggetto dinamico e viene scartato.

## Gate dinamico

Nel file `lidar_ekf_dynamic_gate.py` sono confrontate tre modalita':

- `none`: nessun gate, tutte le misure vengono accettate.
- `static`: gate fisso con soglia chi-quadro al 95% per 2 dimensioni.
- `dynamic`: soglia adattata in funzione di incertezza, velocita' e distanza della misura.

La soglia dinamica usata nella simulazione e':

```text
gate = base_gate
       + gain_velocita * |v|
       + gain_incertezza * sqrt(det(S))
       - gain_range * range
```

Il valore viene limitato tra una soglia minima e massima. L'idea pratica e':

- quando la stima e' piu' incerta o il robot si muove piu' velocemente, il gate puo' essere piu' tollerante;
- per misure lontane, spesso piu' ambigue in ambienti dinamici, il gate viene leggermente ristretto;
- le persone mobili tendono ad avere innovazioni incoerenti con la mappa e vengono scartate.

## Come eseguire la simulazione

Per l'esperimento automatico, dalla cartella del progetto:

```powershell
.\.venv\Scripts\python.exe .\lidar_ekf_dynamic_gate.py
```

Per cambiare seed o durata:

```powershell
.\.venv\Scripts\python.exe .\lidar_ekf_dynamic_gate.py --seed 12 --steps 700
```

Il programma stampa:

- RMSE della posizione;
- errore finale;
- numero di persone mobili accettate per errore;
- numero di landmark statici respinti.

Inoltre salva una figura in:

```text
results/lidar_dynamic_gate_results.png
```

Per la simulazione interattiva con tastiera:

```powershell
.\.venv\Scripts\python.exe .\interactive_lidar_dynamic_gate.py
```

Comandi:

- frecce oppure `WASD`: guida il robot;
- `Spazio`: ferma il robot;
- `R`: resetta la posa stimata dall'EKF vicino alla posa vera.

Nella finestra:

- quadrati grigi: oggetti statici noti della mappa;
- quadrati verdi vuoti: oggetti statici accettati dal gate e usati nell'EKF;
- cerchi arancioni: persone mobili simulate;
- croci rosse: misure classificate come dinamiche e scartate;
- robot blu: posa vera;
- robot magenta: posa stimata dall'EKF.

## File del progetto

- `Unicycle.py`: file base originale con simulazione unicycle, IMU, camera ed EKF.
- `kalman_filter.py`: file base originale con EKF usato dalla simulazione fornita.
- `lidar_ekf_dynamic_gate.py`: nuova simulazione Python per Lidar, persone mobili e gate dinamico.
- `interactive_lidar_dynamic_gate.py`: simulazione attiva guidabile con tastiera.
- `README_progetto.md`: questa traccia descrittiva per relazione e presentazione.
