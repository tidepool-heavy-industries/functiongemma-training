# FunctionGemma Training Data Creation Guide

**For LLMs Generating Fine-Tuning Datasets**

This guide provides everything needed to generate high-quality training data for fine-tuning FunctionGemma 270M (or similar function-calling models).

## Table of Contents
1. [Dataset Format Requirements](#dataset-format-requirements)
2. [Example Structure](#example-structure)
3. [Quality Criteria](#quality-criteria)
4. [Generation Template](#generation-template)
5. [Common Pitfalls](#common-pitfalls)
6. [Validation Checklist](#validation-checklist)

---

## Dataset Format Requirements

### File Format
- **Format:** HuggingFace Dataset (Arrow files) OR JSONL
- **Structure:** Each example is a complete conversation turn
- **Encoding:** UTF-8
- **Size Target:** 800-1000 high-quality examples (sweet spot for 270M models)

### Required Fields
```python
{
    "text": str  # Complete formatted conversation (see below)
}
```

---

## Example Structure

### Complete Example Anatomy

```
<start_of_turn>developer
You are an expert function calling AI assistant. Current date and time given in YYYY-MM-DDTHH:MM:SS format: {TIMESTAMP}
You have access to the following functions:

<start_function_declaration>{FUNCTION_NAME}
{FUNCTION_DESCRIPTION}
Parameters:
  {PARAM_NAME} ({PARAM_TYPE}): {PARAM_DESCRIPTION} (required/optional)
  ...
<end_function_declaration>

[... more function declarations ...]

<end_of_turn>
<start_of_turn>user
{USER_QUERY}
<end_of_turn>
<start_of_turn>model
<start_function_call>
call:{FUNCTION_NAME}{{{PARAM_NAME}:<escape>{PARAM_VALUE}<escape>, ...}}
<end_function_call>
[... more function calls if needed ...]
<end_of_turn>
```

### Concrete Example

```
<start_of_turn>developer
You are an expert function calling AI assistant. Current date and time given in YYYY-MM-DDTHH:MM:SS format: 2026-01-16T14:30:00
You have access to the following functions:

<start_function_declaration>turn_on_flashlight
Turns the flashlight on.
Parameters:
<end_function_declaration>

<start_function_declaration>create_calendar_event
Creates a new calendar event.
Parameters:
  datetime (string): The date and time of the event in the format YYYY-MM-DDTHH:MM:SS. (required)
  title (string): The title of the event. (required)
<end_function_declaration>

<end_of_turn>
<start_of_turn>user
Turn on my flashlight and schedule a meeting called Team Standup for tomorrow at 9am
<end_of_turn>
<start_of_turn>model
<start_function_call>
call:turn_on_flashlight{}
<end_function_call>
<start_function_call>
call:create_calendar_event{datetime:<escape>2026-01-17T09:00:00<escape>,title:<escape>Team Standup<escape>}
<end_function_call>
<end_of_turn>
```

---

## Quality Criteria

### High-Quality Examples Must Have:

1. **Realistic User Queries**
   - Natural language, not robotic
   - ✅ "Turn on my flashlight and schedule a meeting called Team Standup for tomorrow at 9am"
   - ❌ "Execute function turn_on_flashlight and function create_calendar_event with parameters datetime and title"

2. **Accurate Parameter Extraction**
   - User says "tomorrow at 9am" → Model outputs `2026-01-17T09:00:00` (assuming today is 2026-01-16)
   - User says "John Smith" → Model splits to `first_name:<escape>John<escape>,last_name:<escape>Smith<escape>`
   - User says "call me at 555-1234" → Model extracts `phone_number:<escape>555-1234<escape>`

3. **Correct Function Selection**
   - Don't call `turn_flashlight` when the function is `turn_on_flashlight`
   - Don't call `send_email` when user wants to create a contact

4. **Proper Datetime Handling**
   - Format: `YYYY-MM-DDTHH:MM:SS`
   - "tomorrow at 3pm" → calculate based on current timestamp in developer prompt
   - "next Tuesday at 2:30pm" → calculate day and format correctly

5. **Multiple Functions Per Example** (Sometimes)
   - ~30% of examples should have 2+ function calls (parallel calling)
   - User: "Turn on flashlight and create event Gym at 6pm"
   - Model calls BOTH functions

6. **Include Negative Examples** (~10-15%)
   - User: "Hello" → Model responds with text, NO function call
   - User: "Thank you" → Model responds politely, NO function call
   - This prevents "trigger-happy" behavior

---

## Generation Template

### Python Code to Generate Examples

```python
import json
from datetime import datetime, timedelta
import random

# Define your function schemas
FUNCTIONS = [
    {
        "name": "turn_on_flashlight",
        "description": "Turns the flashlight on.",
        "parameters": {}
    },
    {
        "name": "turn_off_flashlight",
        "description": "Turns the flashlight off.",
        "parameters": {}
    },
    {
        "name": "create_calendar_event",
        "description": "Creates a new calendar event.",
        "parameters": {
            "datetime": {"type": "string", "required": True, "description": "The date and time of the event in the format YYYY-MM-DDTHH:MM:SS."},
            "title": {"type": "string", "required": True, "description": "The title of the event."}
        }
    },
    {
        "name": "create_contact",
        "description": "Creates a contact in the phone's contact list.",
        "parameters": {
            "first_name": {"type": "string", "required": True, "description": "The first name of the contact."},
            "last_name": {"type": "string", "required": True, "description": "The last name of the contact."},
            "phone_number": {"type": "string", "required": False, "description": "The phone number of the contact."},
            "email": {"type": "string", "required": False, "description": "The email address of the contact."}
        }
    },
    {
        "name": "send_email",
        "description": "Sends an email.",
        "parameters": {
            "to": {"type": "string", "required": True, "description": "The email address of the recipient."},
            "subject": {"type": "string", "required": True, "description": "The subject of the email."},
            "body": {"type": "string", "required": False, "description": "The body of the email."}
        }
    }
]

def format_function_declaration(func):
    """Format a function as FunctionGemma declaration"""
    decl = f"<start_function_declaration>{func['name']}\n"
    decl += f"{func['description']}\n"

    if func['parameters']:
        decl += "Parameters:\n"
        for param_name, param_info in func['parameters'].items():
            req = "required" if param_info.get('required') else "optional"
            decl += f"  {param_name} ({param_info['type']}): {param_info['description']} ({req})\n"
    else:
        decl += "Parameters:\n"

    decl += "<end_function_declaration>"
    return decl

def format_function_call(func_name, params):
    """Format a function call with parameters"""
    call = f"<start_function_call>\ncall:{func_name}{{"

    if params:
        param_strs = []
        for key, value in params.items():
            param_strs.append(f"{key}:<escape>{value}<escape>")
        call += ",".join(param_strs)

    call += "}\n<end_function_call>"
    return call

def generate_example(
    current_datetime,
    user_query,
    function_calls,  # List of (func_name, params) tuples
    available_functions=None
):
    """Generate a complete training example"""

    if available_functions is None:
        available_functions = FUNCTIONS

    # Build developer prompt
    example = "<start_of_turn>developer\n"
    example += f"You are an expert function calling AI assistant. Current date and time given in YYYY-MM-DDTHH:MM:SS format: {current_datetime.strftime('%Y-%m-%dT%H:%M:%S')}\n"
    example += "You have access to the following functions:\n\n"

    for func in available_functions:
        example += format_function_declaration(func) + "\n\n"

    example += "<end_of_turn>\n"

    # Add user query
    example += f"<start_of_turn>user\n{user_query}\n<end_of_turn>\n"

    # Add model response
    example += "<start_of_turn>model\n"

    if function_calls:
        for func_name, params in function_calls:
            example += format_function_call(func_name, params) + "\n"
    else:
        # Negative example - just text response
        example += "I'm here to help! What would you like me to do?\n"

    example += "<end_of_turn>"

    return {"text": example}

# Example usage:

# Positive example - single function
ex1 = generate_example(
    current_datetime=datetime(2026, 1, 16, 14, 30, 0),
    user_query="Turn on the flashlight",
    function_calls=[
        ("turn_on_flashlight", {})
    ]
)

# Positive example - with parameters
ex2 = generate_example(
    current_datetime=datetime(2026, 1, 16, 14, 30, 0),
    user_query="Schedule a meeting called Team Standup for tomorrow at 9am",
    function_calls=[
        ("create_calendar_event", {
            "datetime": "2026-01-17T09:00:00",
            "title": "Team Standup"
        })
    ]
)

# Positive example - parallel calls
ex3 = generate_example(
    current_datetime=datetime(2026, 1, 16, 14, 30, 0),
    user_query="Turn on my flashlight and schedule Gym for today at 6pm",
    function_calls=[
        ("turn_on_flashlight", {}),
        ("create_calendar_event", {
            "datetime": "2026-01-16T18:00:00",
            "title": "Gym"
        })
    ]
)

# Negative example - no function call
ex4 = generate_example(
    current_datetime=datetime(2026, 1, 16, 14, 30, 0),
    user_query="Hello, how are you?",
    function_calls=None  # No function calls
)

# Save to JSONL
with open("training_data.jsonl", "w") as f:
    for ex in [ex1, ex2, ex3, ex4]:
        f.write(json.dumps(ex) + "\n")
```

---

## Parameter Type Guidelines

### Datetime Parameters
**Format:** `YYYY-MM-DDTHH:MM:SS` (ISO 8601 without timezone)

**Relative time parsing:**
- "tomorrow at 3pm" → Add 1 day, set hour to 15
- "next Tuesday at 2:30pm" → Calculate next Tuesday, set time
- "in 2 hours" → Add 2 hours to current time
- "January 15th at noon" → Parse date, set time to 12:00

**Always calculate relative to the timestamp in the developer prompt!**

### String Parameters with `<escape>` Tokens
All string values MUST be wrapped in `<escape>` tokens:

```
title:<escape>Team Meeting<escape>
first_name:<escape>John<escape>
email:<escape>john@example.com<escape>
```

**Why:** Prevents parser confusion between parameter syntax and content.

### Name Splitting
User says full name → Split into first/last:
- "John Smith" → `first_name:<escape>John<escape>,last_name:<escape>Smith<escape>`
- "Maria Garcia Lopez" → `first_name:<escape>Maria<escape>,last_name:<escape>Garcia Lopez<escape>`
- "Dr. Sarah Chen" → `first_name:<escape>Sarah<escape>,last_name:<escape>Chen<escape>` (drop title)

### Phone Numbers
Accept various formats, normalize to digits with dashes:
- "555-1234" → `phone_number:<escape>555-1234<escape>`
- "555 1234" → `phone_number:<escape>555-1234<escape>`
- "(555) 123-4567" → `phone_number:<escape>555-123-4567<escape>`

### Email Addresses
Keep as-is, validate format:
- Must contain `@` and `.`
- Lowercase recommended

---

## Diversity Guidelines

### Query Variety (Critical!)
Don't generate repetitive queries. Vary:

1. **Phrasing:**
   - ✅ "Turn on the flashlight"
   - ✅ "Switch on my flashlight"
   - ✅ "I need the flashlight on"
   - ✅ "Activate flashlight"

2. **Formality:**
   - Casual: "Turn on my flashlight"
   - Polite: "Could you please turn on the flashlight?"
   - Direct: "Flashlight on"

3. **Complexity:**
   - Simple: "Turn on flashlight" (1 function)
   - Medium: "Turn on flashlight and create event" (2 functions)
   - Complex: "Turn on flashlight, schedule Team Meeting for tomorrow at 9am, and save contact John Smith with phone 555-1234" (3 functions)

### Function Distribution
Aim for balanced distribution across all functions:
- Each function should appear in ~15-20% of examples
- Include combinations of functions
- Don't over-represent simple functions

### Parameter Diversity
Use realistic, varied values:

**Event titles:**
- Team Standup, Client Meeting, Doctor Appointment, Gym, Lunch with Sarah, Project Review, etc.

**Contact names:**
- Use diverse cultural names: John Smith, Maria Garcia, Wei Chen, Fatima Hassan, etc.

**Email subjects:**
- Project Update, Quick Question, Meeting Notes, Follow-up, etc.

**Timestamps:**
- Vary time of day, day of week, relative vs absolute references

---

## Common Pitfalls

### ❌ WRONG: Missing `<escape>` Tokens
```
call:create_calendar_event{datetime:2026-01-17T09:00:00,title:Meeting}
```

### ✅ CORRECT:
```
call:create_calendar_event{datetime:<escape>2026-01-17T09:00:00<escape>,title:<escape>Meeting<escape>}
```

---

### ❌ WRONG: Incorrect Function Name
User says "turn on flashlight" but function is `turn_on_flashlight`:
```
call:turn_flashlight{}
```

### ✅ CORRECT:
```
call:turn_on_flashlight{}
```

---

### ❌ WRONG: Not Splitting Names
```
call:create_contact{first_name:<escape>John Smith<escape>,last_name:<escape><escape>}
```

### ✅ CORRECT:
```
call:create_contact{first_name:<escape>John<escape>,last_name:<escape>Smith<escape>}
```

---

### ❌ WRONG: Inconsistent Datetime Format
```
datetime:<escape>Jan 17, 2026 9:00am<escape>
```

### ✅ CORRECT:
```
datetime:<escape>2026-01-17T09:00:00<escape>
```

---

### ❌ WRONG: No Negative Examples
All 1000 examples call functions → Model becomes "trigger-happy"

### ✅ CORRECT:
Include 100-150 examples where the correct response is NO function call:
- Greetings: "Hello", "Hi there"
- Thanks: "Thank you", "Thanks!"
- Chit-chat: "How are you?", "What's up?"
- Out of scope: "Tell me a joke", "What's the weather?" (if no weather function)

---

## Validation Checklist

Before considering dataset complete, verify:

- [ ] **Format:** Every example follows the exact structure (developer → user → model)
- [ ] **Tokens:** All string parameters use `<escape>` tokens
- [ ] **Function names:** Exactly match the declared function names
- [ ] **Parameters:** All required parameters present, no hallucinated parameters
- [ ] **Datetime format:** All datetimes in `YYYY-MM-DDTHH:MM:SS` format
- [ ] **Name splitting:** Full names properly split into first/last
- [ ] **Diversity:** Varied queries, not repetitive
- [ ] **Negative examples:** 10-15% have no function calls
- [ ] **Parallel calls:** 20-30% have 2+ function calls
- [ ] **Length:** 800-1000 examples total
- [ ] **Quality:** Each example is realistic and useful

---

## Automated Validation Script

```python
import json
import re

def validate_example(example):
    """Validate a training example"""
    errors = []
    text = example['text']

    # Check structure
    if '<start_of_turn>developer' not in text:
        errors.append("Missing developer turn")
    if '<start_of_turn>user' not in text:
        errors.append("Missing user turn")
    if '<start_of_turn>model' not in text:
        errors.append("Missing model turn")

    # Check for function calls
    function_calls = re.findall(r'<start_function_call>\s*call:(\w+)\{([^}]*)\}\s*<end_function_call>', text)

    for func_name, params_str in function_calls:
        # Check parameter format
        if params_str.strip():  # Has parameters
            # Check for escape tokens
            params = re.split(r',(?![^<]*<escape>)', params_str)
            for param in params:
                if ':' in param:
                    key, value = param.split(':', 1)
                    if '<escape>' not in value and value.strip():
                        errors.append(f"Parameter {key.strip()} missing <escape> tokens")

                    # Check datetime format if applicable
                    if 'datetime' in key.lower():
                        # Extract value between escape tokens
                        datetime_match = re.search(r'<escape>([^<]+)<escape>', value)
                        if datetime_match:
                            dt_value = datetime_match.group(1)
                            if not re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', dt_value):
                                errors.append(f"Invalid datetime format: {dt_value}")

    return errors

def validate_dataset(jsonl_path):
    """Validate entire dataset"""
    with open(jsonl_path, 'r') as f:
        examples = [json.loads(line) for line in f]

    print(f"Validating {len(examples)} examples...")

    total_errors = 0
    for i, example in enumerate(examples):
        errors = validate_example(example)
        if errors:
            print(f"\nExample {i}: {len(errors)} error(s)")
            for error in errors:
                print(f"  - {error}")
            total_errors += len(errors)

    if total_errors == 0:
        print("\n✅ All examples valid!")
    else:
        print(f"\n❌ Found {total_errors} total errors")

    return total_errors == 0

# Usage:
validate_dataset("training_data.jsonl")
```

---

## Example Generation Strategies

### Strategy 1: Template-Based
Define query templates and fill in variables:
```python
templates = [
    "Turn on the {device}",
    "Schedule a {event_type} called {title} for {time}",
    "Save contact {name} with phone {phone}",
    "Send email to {recipient} about {subject}"
]
```

### Strategy 2: Compositional
Combine simple queries into complex ones:
```python
simple = ["Turn on flashlight", "Create event Meeting at 3pm"]
complex = " and ".join(simple)
# "Turn on flashlight and create event Meeting at 3pm"
```

### Strategy 3: Synthetic Variation
Take a base query and generate variations:
```python
base = "Turn on the flashlight"
variations = [
    "Switch on my flashlight",
    "I need the flashlight turned on",
    "Could you turn on the flashlight please?",
    "Flashlight on"
]
```

---

## Dataset Size Recommendations

| Model Size | Recommended Examples | Training Time (GTX 1660 SUPER) |
|------------|---------------------|-------------------------------|
| 270M       | 800-1000            | ~90 minutes                   |
| 500M       | 1500-2000           | ~3 hours                      |
| 1.5B       | 3000-5000           | ~8-12 hours                   |

**Sweet spot for 270M:** 900 examples with high quality beats 5000 low-quality examples.

---

## Final Output Format

### Option 1: JSONL File
```jsonl
{"text": "<start_of_turn>developer\n..."}
{"text": "<start_of_turn>developer\n..."}
...
```

### Option 2: HuggingFace Dataset
```python
from datasets import Dataset

examples = [...]  # List of {"text": "..."} dicts
dataset = Dataset.from_list(examples)
dataset.save_to_disk("./my_training_data")
```

### Option 3: Split Train/Test
```python
from sklearn.model_selection import train_test_split

train, test = train_test_split(examples, test_size=0.1, random_state=42)

train_dataset = Dataset.from_list(train)
test_dataset = Dataset.from_list(test)

train_dataset.save_to_disk("./my_training_data_train")
test_dataset.save_to_disk("./my_training_data_test")
```

---

## Quality Over Quantity

Remember:
- **900 high-quality examples > 5000 low-quality examples**
- Each example should teach the model something useful
- Avoid duplicates or near-duplicates
- Focus on edge cases and challenging scenarios
- Include diversity in queries, functions, and parameters

---

## Testing Your Dataset

Before training, manually review:
1. **First 10 examples** - Check formatting
2. **Random 20 examples** - Check diversity
3. **Last 10 examples** - Check consistency
4. **All negative examples** - Ensure they make sense

Run the validation script above to catch systematic errors.

---

## Summary Checklist

Before starting training:
- [ ] Dataset has 800-1000 examples
- [ ] All examples validated (structure, format, tokens)
- [ ] 10-15% negative examples (no function calls)
- [ ] 20-30% parallel function calls
- [ ] Diverse queries (not repetitive)
- [ ] Realistic parameters (names, times, emails)
- [ ] Correct datetime format everywhere
- [ ] All string parameters have `<escape>` tokens
- [ ] Function names match declarations exactly
- [ ] Split into train (90%) and test (10%)

**If all checkboxes pass → Ready to train!** 🚀
