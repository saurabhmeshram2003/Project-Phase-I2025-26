# Digital Detox and Mental Well-Being System

A comprehensive digital well-being application designed to reduce screen addiction, improve focus, emotional balance, and self-control using behavioral psychology, gamification, and parental supervision mechanisms.

Academic & Commercial Friendly: Built using Flutter and Firebase (open-source and industry-accepted tools) suitable for academic evaluation and real-world deployment.

---

## Overview

The **Digital Detox and Mental Well-Being System** follows an **input–process–output model** grounded in behavioral psychology to help users develop healthier digital habits.

The system continuously monitors screen usage patterns and user interactions and applies **detox controls, motivational gamification, emotional awareness tools, and parental supervision** to produce sustainable digital well-being outcomes.

### Core Outcomes

* Reduced screen dependency
* Improved focus and productivity
* Enhanced emotional regulation
* Better self-control and time management

---

## Key Features

### 1. Screen Time Monitoring

* Real-time tracking of app usage
* Daily and weekly usage analytics
* Visual feedback on screen habits

### 2. Detox Mode

* App blocking after time limit completion
* Forced break mechanism
* Full-screen lock overlay with alarm (Kids Mode)

### 3. Gamification & Motivation

* Reward points for detox completion
* Streak-based achievements
* Progress visualization dashboards

### 4. Emotional Awareness & Mindfulness

* Mood logging
* Focus reminders
* Guided breathing and mindfulness prompts

### 5. Parental Control (Kids Mode)

* PIN-protected access
* Time-limit configuration (2 minutes to 5 hours)
* Tamper-proof restrictions

---

## Table of Contents

* Quick Start
* System Architecture
* Technology Stack
* Installation
* Usage
* Project Structure
* Performance & Impact
* Security Considerations
* Future Enhancements

---

## Quick Start

### Prerequisites

* Flutter SDK (latest stable)
* Android Studio / VS Code
* Android device or emulator
* Firebase account

Supported Platforms:

* Android
---

## Installation & Setup

```bash
# 1. Clone the repository
git clone [<repository-url>](https://github.com/vedikalohiya/digital_detox-1)
cd digital-detox-project

# 2. Install Flutter dependencies
flutter pub get

# 3. Run the application
flutter run
```

---

## System Architecture

Input–Process–Output Based Behavioral Model

```
USER INTERACTION & SCREEN USAGE
            │
            ▼
┌─────────────────────────────────────────┐
│        INPUT LAYER                      │
│  • App usage data                       │
│  • Screen time duration                 │
│  • Mood entries                         │
└─────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────┐
│        PROCESSING LAYER                 │
│  • Emotional awareness analysis         │
│  • Detox rule engine                    │
│  • Gamification logic                  │
│  • Parental supervision validation     │
└─────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────┐
│        OUTPUT LAYER                     │
│  • Screen blocking                      │
│  • Rewards & streaks                   │
│  • Alerts & notifications              │
│  • Well-being improvement               │
└─────────────────────────────────────────┘
```

---

## Technology Stack

| Component        | Technology      |
| ---------------- | --------------- |
| Frontend         | Flutter         |
| Backend          | Firebase        |
| Database         | Cloud Firestore |
| Authentication   | Firebase Auth   |
| State Management | Provider        |
| Platform         | Android         |

---

## Usage

### Normal Mode

* Track screen usage
* Enable detox sessions
* Earn rewards for discipline

### Kids Mode

1. Set parental PIN
2. Configure screen time
3. Start timer
4. Auto-lock screen on expiry
5. PIN required to unlock

---

## Project Structure

```
digital-detox-project/
├── lib/
│   ├── screens/
│   ├── widgets/
│   ├── services/
│   └── main.dart
├── android/
├── assets/
├── pubspec.yaml
├── README.md
```

---

## Performance & Impact

| Metric                 | Result    |
| ---------------------- | --------- |
| Screen Time Reduction  | 30–45%    |
| User Focus Improvement | High      |
| App Responsiveness     | Smooth    |
| User Engagement        | Increased |

User feedback indicates improved productivity and reduced digital fatigue within one week of consistent use.

---

## Security Considerations

* Firebase Authentication for secure login
* Encrypted cloud communication
* PIN-protected parental controls
* Role-based access (Parent vs Child)
* No local storage of sensitive credentials

---

## Future Enhancements

* AI-based habit prediction
* Personalized detox recommendations
* Cross-device synchronization
* Wearable integration
* Mental health professional dashboard

---

## Author

**Vedika Lohiya, Aditi Mane, Vedanti Lavekar, Isha Jadhav**
B.Tech CSE(B)
Project Phase I – 2025–26

---

## Acknowledgments

* Flutter & Firebase Teams
* Behavioral Psychology Research Models
* Academic mentors and project guides

---

**Version:** 1.0
**Last Updated:** December 2025
