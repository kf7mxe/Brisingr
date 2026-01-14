# Lightning Time Tracker Skill

Shell-based Claude skill for interacting with Lightning Time Tracker.

## Setup

1. Copy the config template:
   ```bash
   cd ~/.claude/skills/lightning-time-tracker
   cp config.example config
   ```

2. Edit `config` and add your Lightning Time Tracker session token:
   ```bash
   TOKEN=your_actual_session_token_here
   BASE_URL=https://lk.api.lightningtime.com
   ```

3. The skill is now ready to use! Claude will automatically discover it.

## Features

- List projects, tasks, and time entries
- Start and stop time tracking
- Auto-stops existing timers when starting a new one
- Defaults to showing today's time entries

## Dependencies

- `curl` - HTTP requests (pre-installed on macOS)
- `jq` - JSON parsing (install with: `brew install jq`)

## Scripts

All scripts output JSON for Claude to parse:

- `scripts/list-projects.sh` - Get all projects
- `scripts/list-tasks.sh [project_id]` - Get tasks, optionally filtered
- `scripts/list-entries.sh [start] [end]` - Get time entries (defaults to today)
- `scripts/active-timer.sh` - Check for running timer
- `scripts/start-timer.sh <task_id> [description]` - Start timer
- `scripts/stop-timer.sh` - Stop active timer

## Example Usage

Ask Claude:
- "What projects am I working on?"
- "Show me my tasks for project X"
- "Start tracking time for task 12345"
- "What have I logged today?"
- "Stop my timer"

Claude will use the appropriate scripts and format the results for you.
