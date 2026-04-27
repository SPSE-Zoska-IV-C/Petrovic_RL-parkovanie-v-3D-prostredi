# AI Auto-Parking - Curriculum Learning

Školský projekt zameraný na autonómne parkovanie auta v 3D simulácii pomocou reinforcement learningu (SAC), Unity ML-Agents a vlastného reward systému v Pythone.

## O projekte

Cieľ projektu je natrénovať agenta, ktorý dokáže samostatne zaparkovať v simulovanom prostredí. Tréning prebieha cez Unity simuláciu (pozorovania + akcie) a Python tréningový pipeline (SAC, wrapping prostredia, reward shaping).

### Ako funguje curriculum learning

Curriculum learning je stratégia, pri ktorej sa agent učí od jednoduchších scenárov k náročnejším. Vďaka tomu sa rýchlejšie stabilizuje učenie a znižuje sa náhodné správanie v zložitých situáciách.

## Technológie

- Unity + ML-Agents (simulácia a agent v C#)
- Python 3.10+
- Stable-Baselines3 (SAC)
- Gymnasium
- NumPy
- TensorBoard

## Inštalácia

### Požiadavky

- Python 3.10+
- Unity 2022.x+ s ML-Agents
- Git

### 1) Klonovanie repozitára

```bash
git clone https://github.com/SPSE-Zoska-IV-C/Petrovic_RL-parkovanie-v-3D-prostredi.git
cd Petrovic_RL-parkovanie-v-3D-prostredi
```

### 2) Inštalácia Python závislostí

```bash
pip install -r requirements.txt
```

## Spustenie tréningu

Spúšťanie tréningu je v priečinku `python/`. Pred spustením je potrebné mať Unity build prostredia (Windows `.exe` alebo Linux `.x86_64`) a správnu cestu k nemu v skripte/argumentoch.

Príklad:

```bash
cd python
python train.py
```

TensorBoard logy je možné sledovať cez:

```bash
tensorboard --logdir ./logs
```

## Aktuálna štruktúra projektu

```text
Petrovic_RL-parkovanie-v-3D-prostredi/
├── ENV/
│   └── AutoMaturitaEasy/
│       └── Assets/
│           └── Scripts/
│               ├── Camera.cs
│               ├── CarAgent.cs
│               ├── CarController.cs
│               ├── CarCrashHandler.cs
│               ├── LidarSensor.cs
│               ├── ParkingManager.cs
│               ├── ParkingSpot.cs
│               ├── TEST.cs
│               └── WheelScript.cs
├── models/
├── python/
│   ├── reward_calc.py
│   ├── sac_t.py
│   ├── train.py
│   ├── wrapper.py
│   └── tests/
│       ├── debug.py
│       ├── fatal.py
│       ├── phases.py
│       ├── testik.py
│       └── try_connect.py
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Kredity

Projekt bol vyvinutý ako školský projekt na **SPSE Zoška** (IV.C, školský rok 2024/2025).

Použité open-source technológie:

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents)
- [Gymnasium](https://gymnasium.farama.org/)
- [NumPy](https://numpy.org/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)

Vedúci práce: Oliver Halaš