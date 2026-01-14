This skill helps you interact with Lightning Time Tracker to manage projects, tasks, and time tracking.

## Available Operations

### View Information
- **List projects**: Get all available projects
- **List tasks**: Get tasks, optionally filtered by project
- **List time entries**: View time entries (defaults to today)
- **Check active timer**: See if a timer is currently running

### Time Tracking
- **Start timer**: Begin tracking time for a task (auto-stops any active timer)
- **Stop timer**: Stop the currently running timer

## Scripts

All scripts are in `~/.claude/skills/lightning-time-tracker/scripts/` and output JSON:

- `list-projects.sh` - Returns array of all projects
- `list-tasks.sh [project_id]` - Returns array of tasks, optionally filtered by project
- `list-entries.sh [start_date] [end_date]` - Returns time entries (defaults to today if dates not provided)
- `active-timer.sh` - Returns array of active timers (empty if none)
- `start-timer.sh <task_id> [description]` - Starts timer and returns created timer object
- `stop-timer.sh` - Stops active timer(s) and returns stopped timer object(s)

## Usage

Call scripts directly using the Bash tool. Parse JSON responses to present information to the user in a friendly format.

Examples:
- User: "What projects am I working on?" → Run `list-projects.sh`, parse and display
- User: "Start tracking time for task X" → Run `start-timer.sh <task_id>`
- User: "What have I logged today?" → Run `list-entries.sh`, summarize results
- User: "Stop my timer" → Run `stop-timer.sh`

## Setup

User must copy `config.example` to `config` and add their Lightning Time Tracker session token before using this skill.
