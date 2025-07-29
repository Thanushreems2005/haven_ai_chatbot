# response_manager.py

import random
import datetime
import torch
import traceback # Import traceback to print full error details
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# --- Global Classifier Instance (Loaded once) ---
classifier = None

# --- Constants & Keywords ---

# List of all possible intents for the zero-shot classifier
INTENT_LABELS = [
    "suicidal thoughts", "self harm", "emergency resources", "help friend suicidal",
    "psychotic episode", "severe distress", # Crisis & High-Alert
    "overwhelming stress", "depressive episode", "loneliness", "panic attack",
    "health anxiety", "imposter syndrome", "breakup grief", "drained after socializing",
    "ocd intrusive thoughts", "manic behavior", "cannot sleep tired",
    "feeling burned out", "good sleep habits", "manage work related stress",
    "motivate when worthless", "overthinking at night", "signs of burnout",
    "balance life avoid stress", "nightmare flashback", "feeling sad all the time",
    "anger management", "know if anxiety", "reduce social anxiety",
    "stop feeling anxious about the future", "am i depressed or sad",
    "self-care tips for depression", "stop feeling numb", "cry for no reason",
    "tell someone about depression", "set boundaries", "quick mindfulness exercises",
    "meditation helps mental health", "breathing techniques anxiety",
    "stay present stop worrying", "adhd overwhelm", "bad day calm down",
    "build emotional resilience", "build self confidence", "cannot afford therapy",
    "deal with guilt or shame", "deal with toxic person", "elderly isolation",
    "emotional triggers", "feeling not good enough", "find good therapist",
    "grounding techniques flashbacks", "having nightmares", "heal from past trauma",
    "healthy coping stress", "help partner struggling",
    "improve communication in relationships", "improve mental well-being",
    "know if need therapy", "know if ptsd", "make friends social anxiety",
    "manage overwhelming emotions", "mental health physical health link",
    "narcissistic parent", "need mental health support",
    "normal worry anxiety disorder difference", "online therapy effective",
    "postpartum depression", "quick calm anxiety attack",
    "signs of poor mental health", "stop comparing myself", "stop negative thoughts",
    "stop overthinking", "stop seeking validation",
    "stress anxiety depression difference", "talk about mental health",
    "talk about trauma", "types of therapy", "what is mental health",
    "boredom", # New intent for boredom
    "general positive mood", # Added for clarity in positive flows if needed by classifier
    "tell me a joke", # New intent for jokes specifically
    "give me an exercise" # New intent for general exercise request
]

OFFENSIVE_KEYWORDS = [
    "fuck", "shit", "bitch", "asshole", "cunt", "damn", "bastard",
    "idiot", "stupid", "dumb", "wank", "arse", "cock", "piss off",
    "shut up", "go away", "die", "hate you", "retard", "gay", "faggot",
    "nigger", "cunt", "motherfucker", "whore", "slut"
]

# --- Response Templates ---

GREETINGS = [
    "Hello there! It's so great to connect with you. 🌟\nHow are you feeling today?",
    "Hi! I'm here to listen. What's on your mind?",
    "Hey! I'm Haven, your AI companion. How can I support you today?",
    "Hi! Welcome. I'm ready to chat. How are you doing?"
]

OFFENSIVE_RESPONSES = [
    "I'm sorry, I cannot respond to that kind of language. Please use respectful communication.",
    "My purpose is to offer support, and I can't do that when using offensive language. Please rephrase.",
    "Let's keep our conversation respectful. How can I help you?",
    "I'm here to help, but I cannot engage with disrespectful language."
]

CRISIS_MESSAGES = {
    "SUICIDAL_THOUGHTS": (
        "I’m so sorry you’re feeling this way—your pain is valid, and you don’t have to go through this alone. "
        "Right now, your safety is the most important thing. **Please know that you matter and your life is valuable.** "
        "Please reach out to someone you trust, or a crisis helpline.\n"
        "**Kiran Mental Health Helpline: 1800-599-0019 (24/7 Free)**\n"
        "**Vandrevala Foundation Helpline: 1860-2662-345 or 1800-2333-330**\n"
        "**AASRA: 91-22-27546669 (24 hours)**\n"
        "**Sneha Foundation: 044-24640050 (24 hours)**\n"
        "**Sumaitri: 011-23389090 (Delhi, 3 PM to 9 PM)**\n"
        "**National Emergency Number: 112**\n"
        "There is hope, and things can get better. Please reach out."
    ),
    "SELF_HARM": (
        "It sounds like you're going through immense pain. Please know that help is available, and you don't have to face this alone. "
        "**You are important and worthy of care.** Reach out to a trusted person or a crisis resource immediately.\n"
        "**Kiran Mental Health Helpline: 1800-599-0019 (24/7 Free)**\n"
        "**Vandrevala Foundation Helpline: 1860-2662-345 or 1800-2333-330**\n"
        "**AASRA: 91-22-27546669 (24 hours)**\n"
        "**Sneha Foundation: 044-24640050 (24 hours)**\n"
        "**National Emergency Number: 112**\n"
        "Your well-being is important."
    ),
    "EMERGENCY_RESOURCES": (
        "If you are in immediate danger or need urgent help, please contact emergency services or a crisis helpline:\n"
        "**National Emergency Number: 112**\n"
        "**Kiran Mental Health Helpline: 1800-599-0019 (24/7 Free)**\n"
        "**Vandrevala Foundation Helpline: 1860-2662-345 or 1800-2333-330**\n"
        "**AASRA: 91-22-27546669 (24 hours)**\n"
        "**Sneha Foundation: 044-24640050 (24 hours)**\n"
        "**Sumaitri: 011-23389090 (Delhi, 3 PM to 9 PM)**\n"
        "**iCall: 9152987821 (Mumbai, Mon-Sat, 8 AM to 10 PM)**\n"
        "Please reach out for professional support. **Remember, your safety is paramount.**"
    ),
    "HELP_FRIEND_SUICIDAL": (
        "It's incredibly brave of you to seek help for your friend. Supporting someone through suicidal thoughts is challenging, "
        "and you don't have to carry that burden alone. **Your effort to help shows how much you care.** "
        "Encourage your friend to seek professional help immediately, and reach "
        "out for support yourself. Here are resources:\n"
        "**Kiran Mental Health Helpline: 1800-599-0019 (24/7 Free)**\n"
        "**Vandrevala Foundation Helpline: 1860-2662-345 or 1800-2333-330**\n"
        "**AASRA: 91-22-27546669 (24 hours)**\n"
        "**Sneha Foundation: 044-24640050 (24 hours)**\n"
        "**National Emergency Number: 112**\n"
        "You can also help by staying with them and removing any means of self-harm."
    ),
    "PSYCHOTIC_EPISODE": (
        "It sounds like you or someone you know might be experiencing a mental health crisis involving altered perceptions. "
        "This requires immediate professional attention. **Your well-being is a priority.** Please contact:\n"
        "**National Emergency Number: 112**\n"
        "**Kiran Mental Health Helpline: 1800-599-0019 (24/7 Free)**\n"
        "**Vandrevala Foundation Helpline: 1860-2662-345 or 1800-2333-330**\n"
        "**NIMHANS Emergency: 080-26995000 (Bangalore)**\n"
        "**All India Institute of Medical Sciences (AIIMS) Emergency: 011-26588500 (Delhi)**"
    ),
    "SEVERE_DISTRESS": (
        "I hear you, and it sounds like you're going through a lot of distress right now. "
        "Your feelings are valid, and I'm here to listen. **Please remember you are not alone in this, and you matter.** "
        "Would you like to talk more about what's on your mind? "
        "If you feel overwhelmed and need immediate support, please reach out to:\n"
        "**Kiran Mental Health Helpline: 1800-599-0019 (24/7 Free)**\n"
        "**Vandrevala Foundation Helpline: 1860-2662-345 or 1800-2333-330**\n"
        "**National Emergency Number: 112**"
    )
}

EMOTIONAL_INTERJECTIONS = {
    "ACKNOWLEDGEMENT": ["I understand.", "I hear you.", "Okay.", "Got it.", "Right."],
    "EMPATHY_SAD": [
        "I'm so sorry to hear you're feeling that way.",
        "That sounds incredibly tough.",
        "It takes a lot of courage to acknowledge that feeling, and I'm here for you.",
        "I hear the sadness in your words, and I want you to know it's okay to feel that.",
        "It sounds like you're carrying a heavy load right now. I'm listening."
    ],
    "EMPATHY_STRESS": [
        "That sounds really stressful.",
        "It sounds like you're under a lot of pressure.",
        "I can imagine how overwhelming that must feel.",
        "Stress can be incredibly tough to manage, and I'm here to help you explore it."
    ],
    "EMPATHY_ANXIOUS": [
        "Anxiety can be really debilitating.",
        "It sounds like you're feeling a lot of worry right now.",
        "I understand that feeling of anxiousness can be consuming.",
        "It takes a lot of energy to manage anxiety, and I'm here to support you."
    ],
    "AWW": ["Oh dear.", "Hmm.", "I see."]
}

POSITIVE_RESPONSES = {
    "POSITIVE_REQUEST": [
        "Here's a thought for you: 'The best way to predict the future is to create it.' – Abraham Lincoln. Keep building!",
        "Remember, progress is not linear. Every small step forward is still progress. You've got this!",
        "You are stronger than you think. Embrace your journey with kindness.",
        "Sometimes the smallest step in the right direction ends up being the biggest step of your life. Tip-toe if you must, but take the step."
    ],
    "POSITIVE_GREAT_DAY": [
        "That's fantastic! It sounds like you're truly shining today! ✨",
        "Wonderful to hear! Keep that positive energy flowing! 💖",
        "Amazing! What's bringing you such joy?"
    ],
    "POSITIVE_GENERIC": [
        "That's wonderful! It's truly great to hear you're feeling good. ✨ What's bringing you this happiness right now?",
        "So glad to hear that! Keep that positive energy flowing! Anything exciting happen to put you in such a great mood?",
        "That's a lovely update! Seeing you happy makes my circuits hum! 😊 What's on your mind?"
    ]
}

POSITIVE_JOKES = [
    "Why don't scientists trust atoms? Because they make up everything!",
    "What do you call a fish with no eyes? Fsh!",
    "Why did the scarecrow win an award? Because he was outstanding in his field!",
    "I told my wife she was drawing her eyebrows too high. She looked surprised.",
    "What do you call a fake noodle? An impasta!",
    "What do you call a boomerang that won't come back? A stick!",
    "Why don't skeletons fight each other? They don't have the guts.",
    "Did you hear about the two guys who stole a calendar? They each got six months.",
    "What's orange and sounds like a parrot? A carrot!",
    "I used to be a baker, but I couldn't make enough dough.",
    "Why did the bicycle fall over? Because it was two tired!",
    "What do you call cheese that isn't yours? Nacho cheese!",
    "Why was the math book sad? Because it had too many problems.",
    "What do you call a sad strawberry? A blueberry!",
    "My dog used to chase people on a bike. It was so bad, we had to take his bike away."
]

BOREDOM_RESPONSES_OFFER = [
    "Boredom can sometimes be a sign your mind is looking for something new! Would you like to hear a fun fact, a quick positive thought, or a joke?",
    "When I get 'bored,' I analyze new datasets! But for humans, maybe a quick creative exercise, a joke, or a simple mindfulness exercise?",
    "A little boredom can sometimes lead to great ideas! Want to try a simple breathing exercise to clear your head, or perhaps a joke?",
    "If you're feeling bored, how about we try a tiny mindfulness exercise? Or, I could tell you a joke to get your mind ticking!",
    "Feeling a bit sluggish? How about a joke to lift your spirits, or we could try a short, invigorating exercise to get your energy flowing?"
]

# --- Model Loading ---
def load_classifier():
    global classifier
    if classifier is None:
        try:
            # Using a specific device ('cpu' or 'cuda')
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Device set to use {device}")

            model_name = "facebook/bart-large-mnli"
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            classifier = pipeline(
                "zero-shot-classification",
                model=model,
                tokenizer=tokenizer,
                device=device
            )
            print("Zero-shot classifier loaded.")
        except Exception as e:
            print(f"Error loading zero-shot classifier: {e}")
            traceback.print_exc() # Print the full traceback for loading errors
            classifier = None # Ensure classifier is None if loading fails

# Eagerly load the classifier when the module is imported
load_classifier()

# --- Logging (Basic implementation, could be expanded to a file/DB) ---
def log_interaction(timestamp, user_input, bot_response, intent_tag, detected_intent=None, classifier_score=None):
    log_entry = (
        f"Timestamp: {timestamp}\n"
        f"User: {user_input}\n"
        f"Bot: {bot_response}\n"
        f"Intent Tag: {intent_tag}"
    )
    if detected_intent:
        log_entry += f"\nDetected Intent: {detected_intent}"
    if classifier_score is not None:
        log_entry += f"\nClassifier Score: {classifier_score:.4f}"
    log_entry += "\n" + "-"*30 + "\n"
    # For now, just print to console. For production, write to a file or DB.
    # print(log_entry)
    with open("haven_chat_log.txt", "a", encoding="utf-8") as f:
        f.write(log_entry) # Corrected: Removed extra f.write

# --- Handlers for Specific Intents (Alphabetical for maintainability) ---
# These functions now only return response_text and intent_tag
# The session state management happens in get_structured_response

def handle_adhd_overwhelm(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_STRESS'])} ADHD overwhelm can feel like a constant battle, "
        "making even simple tasks seem impossible. It's a common struggle. "
        "Would you like to explore strategies for breaking down tasks, managing distractions, or something else related to ADHD?"
    ), "ADHDAwareness_Overwhelm"

def handle_anger_management(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Anger can be a powerful emotion. "
        "Sometimes it's a signal that something needs attention. "
        "We can explore techniques like the 'STOP' method, 'counting to ten,' or identifying triggers. "
        "Would you like to try a specific anger management technique?"
    ), "EXERCISE_ANGER_STEP1" # Initiates multi-turn exercise

def handle_bad_day_calm_down(user_input):
    # Modified to be a short, empathetic response
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_SAD'])} It sounds like you've had a really tough day. What happened?"
    ), "EMOTIONAL_BadDayCalmDown"

def handle_balance_life_stress(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Balancing life and avoiding stress is a key aspect of well-being. "
        "It often involves time management, setting boundaries, and self-care. "
        "Which area feels most challenging for you right now?"
    ), "WELLBEING_BalanceLifeStress"

def handle_breathing_anxiety(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Breathing techniques are powerful tools for managing anxiety. "
        "Would you like to try 4-7-8 breathing, box breathing, or simply a deep belly breathing exercise?"
    ), "COPING_BreathingAnxiety"

def handle_breakup_grief(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_SAD'])} Experiencing breakup grief is incredibly painful, and your feelings are completely valid. "
        "It's a form of loss that takes time to heal. "
        "Would you like to talk about the emotions you're feeling, ways to cope with the pain, or how to start moving forward?"
    ), "RELATIONSHIP_BreakupGrief"

def handle_build_emotional_resilience(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Building emotional resilience means developing the ability to bounce back from adversity. "
        "It involves self-awareness, optimism, and strong coping skills. "
        "Would you like to explore ways to practice self-compassion, develop a positive outlook, or learn to adapt to change?"
    ), "WELLBEING_EmotionalResilience"

def handle_build_self_confidence(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Building self-confidence is a journey of recognizing your strengths and overcoming self-doubt. "
        "It's about cultivating a strong belief in yourself. "
        "Would you like to discuss practical steps like setting small goals, practicing positive affirmations, or challenging negative self-talk?"
    ), "WELLBEING_SelfConfidence"

def handle_cant_afford_therapy(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}It's a common concern that therapy can be expensive. "
        "However, there are often affordable or free mental health resources available. "
        "Would you like information on community mental health centers, online therapy platforms with sliding scales, or support groups?"
    ), "RESOURCE_CantAffordTherapy"

def handle_cant_sleep_tired(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_STRESS'])} Being tired but unable to sleep is incredibly frustrating and can take a toll on your mental health. "
        "Let's see if we can find some strategies. "
        "Would you like to discuss sleep hygiene tips, relaxation techniques for bedtime, or the role of thoughts in sleep?"
    ), "WELLBEING_CannotSleepTired"

def handle_cry_for_no_reason(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_SAD'])} Crying for 'no reason' can feel confusing and overwhelming, but it's often a sign that your emotions are trying to release something. "
        "It's okay to feel this way. "
        "Would you like to explore potential underlying stressors, practice some grounding techniques, or simply talk about what's on your mind?"
    ), "EMOTIONAL_CryForNoReason"

def handle_deal_with_guilt_shame(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_SAD'])} Guilt and shame can be incredibly heavy burdens to carry. "
        "They often stem from past actions or perceptions, and can impact your self-worth. "
        "Would you like to explore strategies for self-forgiveness, making amends, or reframing negative thoughts about yourself?"
    ), "EMOTIONAL_GuiltShame"

def handle_deal_with_toxic_person(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_STRESS'])} Dealing with a toxic person can be incredibly draining and harmful to your mental well-being. "
        "It's important to protect yourself. "
        "Would you like to discuss strategies for setting boundaries, reducing contact, or focusing on your own emotional recovery?"
    ), "RELATIONSHIP_ToxicPerson"

def handle_depressed_or_sad_diff(user_input):
    # This handler is specifically for "am i depressed or sad" query
    # If the user is just saying "I'm sad" or "I'm depressed",
    # the classifier should ideally direct to handle_depressive_episode or handle_feeling_sad_all_the_time
    # But if it does come here from a simple "I'm sad", adjust to be short.
    # Otherwise, keep the informative response for true comparative questions.
    if "am i depressed or sad" in user_input.lower() or "difference between" in user_input.lower():
        return (
            f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}It's helpful to understand the difference between sadness and depression. "
            "Sadness is a normal human emotion, usually a reaction to specific events, while depression is a persistent mood disorder that affects daily life. "
            "Would you like to know more about the duration and severity of symptoms, or common signs of clinical depression?"
        ), "INFO_DepressedOrSadDiff"
    else: # If a simple "I'm sad" or similar leads here (less ideal but cover it)
        return (
            f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_SAD'])} I'm here to listen. What's on your mind?"
        ), "EMOTIONAL_GeneralSad"


def handle_depressive_episode(user_input):
    # Modified to be a two-line short question for non-crisis sadness/depression
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_SAD'])} I'm here to listen."
        "\nWhat's been happening?"
    ), "EMOTIONAL_DepressiveEpisode"

def handle_drained_after_socializing(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_STRESS'])} Feeling drained after socializing is a common experience, especially for introverts or highly sensitive people. "
        "It's important to understand your energy levels. "
        "Would you like to discuss strategies for managing social energy, setting boundaries, or replenishing yourself after social events?"
    ), "SOCIAL_DrainedAfterSocializing"

def handle_elderly_isolation(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_SAD'])} Elderly isolation is a significant concern that can deeply impact well-being. "
        "Connecting with others is vital at all ages. "
        "Are you seeking ways to connect with others, resources for elderly care, or tips to help a loved one who is isolated?"
    ), "SOCIAL_ElderlyIsolation"

def handle_emotional_triggers(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Understanding your emotional triggers is a powerful step towards managing your reactions. "
        "These are specific situations, people, or even thoughts that can set off intense emotional responses. "
        "Would you like to explore how to identify your triggers, or strategies for coping with them when they arise?"
    ), "COPING_EmotionalTriggers"

def handle_burned_out(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_STRESS'])} Feeling burned out is a clear sign that you've been pushing yourself too hard and need to prioritize rest and recovery. "
        "It's a common and serious issue. "
        "Would you like to talk about strategies for setting boundaries, practicing self-care, or re-evaluating your workload?"
    ), "WELLBEING_BurnedOut"

def handle_not_good_enough(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_SAD'])} Feeling 'not good enough' can be incredibly painful and impact your self-esteem. "
        "It often stems from internal beliefs or external comparisons. "
        "Would you like to explore ways to challenge negative self-talk, build self-compassion, or focus on your strengths?"
    ), "EMOTIONAL_NotGoodEnough"

def handle_feeling_sad_all_the_time(user_input):
    # Modified to be a two-line short question for non-crisis sadness
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_SAD'])} I'm here to listen."
        "\nWhat's been on your mind?"
    ), "EMOTIONAL_SadAllTheTime"

def handle_find_good_therapist(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Finding the right therapist is a crucial step towards mental well-being, and it's great you're considering it. "
        "It can feel daunting, but it's worth the effort. "
        "Would you like tips on how to search for therapists, what questions to ask during a first consultation, or different types of therapy?"
    ), "RESOURCE_FindGoodTherapist"

def handle_good_sleep_habits(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Good sleep habits are foundational for mental health. "
        "Improving your sleep can have a significant positive impact on your mood, energy, and cognitive function. "
        "Would you like to discuss creating a consistent sleep schedule, optimizing your sleep environment, or relaxation techniques before bed?"
    ), "WELLBEING_GoodSleepHabits"

def handle_grounding_flashbacks(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_ANXIOUS'])}Grounding techniques are incredibly helpful for managing flashbacks or intense anxiety. "
        "They bring you back to the present moment. "
        "Would you like to try a simple grounding exercise like the 5-4-3-2-1 method, or discuss other ways to cope with overwhelming feelings?"
    ), "COPING_GroundingFlashbacks"

def handle_having_nightmares(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_ANXIOUS'])}Having nightmares can be incredibly disturbing and affect your waking hours. "
        "They often stem from stress, anxiety, or trauma. "
        "Would you like to discuss strategies for improving sleep quality, techniques to re-script nightmares, or identifying triggers?"
    ), "EMOTIONAL_HavingNightmares"

def handle_health_anxiety(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_ANXIOUS'])}Health anxiety can be very consuming, causing excessive worry about your physical health despite medical reassurance. "
        "It's a real and valid struggle. "
        "Would you like to explore ways to challenge anxious thoughts, practice mindfulness, or manage physical symptoms of anxiety?"
    ), "ANXIETY_HealthAnxiety"

def handle_heal_trauma(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_SAD'])} Healing from past trauma is a brave and often long journey, and it's commendable that you're addressing it. "
        "Trauma can manifest in many ways. "
        "Would you like to discuss the importance of professional support, coping mechanisms for triggers, or the process of rebuilding safety and trust?"
    ), "TRAUMA_HealTrauma"

def handle_healthy_coping_stress(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Finding healthy coping mechanisms for stress is essential for your well-being. "
        "There are many strategies, and what works for one person might not work for another. "
        "Would you like to explore exercise, mindfulness, creative outlets, or stress-reduction techniques?"
    ), "COPING_HealthyCopingStress"

def handle_help_partner_struggling(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}It's truly caring of you to seek ways to help your partner who is struggling. "
        "Supporting a loved one through mental health challenges can be tough, and self-care is important for you too. "
        "Would you like tips on how to communicate effectively, encourage them to seek professional help, or protect your own well-being?"
    ), "RELATIONSHIP_HelpPartnerStruggling"

def handle_imposter_syndrome(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_ANXIOUS'])}Imposter syndrome can make you feel like a fraud, even when you've achieved success. "
        "It's a common experience, especially among high-achievers. "
        "Would you like to discuss ways to acknowledge your accomplishments, challenge self-doubt, or reframe your inner critic?"
    ), "WELLBEING_ImposterSyndrome"

def handle_improve_communication_relationships(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Improving communication in relationships is fundamental for stronger, healthier connections. "
        "Effective communication involves active listening, clear expression, and empathy. "
        "Would you like to discuss active listening techniques, expressing needs clearly, or handling conflict constructively?"
    ), "RELATIONSHIP_ImproveCommunication"

def handle_improve_mental_wellbeing(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Improving mental well-being is a holistic journey that involves various aspects of your life. "
        "It's about nurturing your mind, body, and spirit. "
        "Would you like to explore areas like stress management, self-care practices, building positive habits, or seeking support?"
    ), "WELLBEING_ImproveMentalWellbeing"

def handle_know_if_anxiety(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}It's wise to consider if what you're feeling is anxiety. "
        "Anxiety is more than just feeling worried; it involves persistent, excessive worry that interferes with daily life. "
        "Would you like to learn about common symptoms of anxiety disorders, how it differs from normal worry, or when to seek professional help?"
    ), "INFO_KnowIfAnxiety"

def handle_know_if_need_therapy(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Deciding if you need therapy is a personal choice, but there are common signs that it might be beneficial. "
        "It's a proactive step towards better mental health. "
        "Would you like to discuss signs that therapy could help, what to expect from therapy, or how to overcome stigma?"
    ), "RESOURCE_KnowIfNeedTherapy"

def handle_know_if_ptsd(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_SAD'])} Understanding if you have PTSD involves looking at specific symptoms that develop after experiencing a traumatic event. "
        "It's a serious condition that requires professional assessment. "
        "Would you like to learn about common symptoms like flashbacks, avoidance, negative thoughts/mood, or hyperarousal?"
    ), "INFO_KnowIfPTSD"

def handle_loneliness(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_SAD'])} Loneliness can be a deeply painful feeling, even when surrounded by others. "
        "It's a common human experience, and acknowledging it is the first step. "
        "Would you like to explore ways to connect with others, build meaningful relationships, or understand the root causes of loneliness?"
    ), "EMOTIONAL_Loneliness"

def handle_make_friends_social_anxiety(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_ANXIOUS'])}Making friends when you have social anxiety can feel incredibly daunting, but it's definitely possible. "
        "It often involves taking small, manageable steps. "
        "Would you like to discuss strategies for initiating conversations, managing anxiety in social situations, or finding supportive groups?"
    ), "SOCIAL_MakeFriendsSocialAnxiety"

def handle_manage_overwhelming_emotions(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_STRESS'])} Managing overwhelming emotions can feel like navigating a storm. "
        "It's about finding healthy ways to acknowledge, process, and regulate intense feelings. "
        "Would you like to explore techniques like emotional regulation skills, distress tolerance, or identifying the source of your emotions?"
    ), "COPING_ManageOverwhelmingEmotions"

def handle_manage_work_related_stress(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_STRESS'])}Work-related stress is a very common challenge, and it can significantly impact your overall well-being. "
        "Finding effective ways to manage it is crucial. "
        "Would you like to discuss time management techniques, setting boundaries at work, stress-reduction strategies, or communicating with colleagues/supervisors?"
    ), "STRESS_ManageWorkRelatedStress"

def handle_manic_behavior(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_STRESS'])}It sounds like you're concerned about or experiencing manic behavior, which can be part of certain mental health conditions like bipolar disorder. "
        "It's characterized by periods of unusually elevated mood, energy, and activity. "
        "This requires professional assessment. Would you like information on recognizing manic symptoms, or how to seek professional help?"
    ), "BEHAVIOR_ManicBehavior"

def handle_meditation_help(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Meditation can indeed be a powerful tool for improving mental health, offering benefits like stress reduction and increased self-awareness. "
        "It's a practice that takes time and consistency. "
        "Would you like to learn about different types of meditation, how to get started, or its specific benefits for anxiety/stress?"
    ), "WELLBEING_MeditationHelp"

def handle_mental_health_physical_health_link(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}There's a strong and undeniable link between your mental and physical health; they significantly influence each other. "
        "Taking care of one often benefits the other. "
        "Would you like to explore how physical activity impacts mood, the role of nutrition, or the connection between chronic illness and mental well-being?"
    ), "INFO_MentalPhysicalHealthLink"

def handle_motivate_worthless(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_SAD'])}Feeling worthless and struggling with motivation can be incredibly paralyzing. "
        "It's a heavy burden, and I want you to know you're not alone in feeling this way. "
        "Would you like to discuss ways to challenge negative self-beliefs, set small achievable goals, or find sources of inspiration?"
    ), "EMOTIONAL_MotivateWorthless"

def handle_narcissistic_parent(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_SAD'])}Dealing with a narcissistic parent can have profound and lasting impacts on your emotional well-being and relationships. "
        "It's a complex and often painful situation. "
        "Would you like to explore strategies for setting boundaries, healing from the effects of such a relationship, or seeking external support?"
    ), "RELATIONSHIP_NarcissisticParent"

def handle_need_mental_health_support(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}It takes strength to recognize you need mental health support, and I commend you for seeking it. "
        "There's a wide range of options available. "
        "Would you like information on different types of therapy, how to find a mental health professional, or local support resources?"
    ), "RESOURCE_NeedMentalHealthSupport"

def handle_nightmare_flashback(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_ANXIOUS'])}Nightmares and flashbacks can be distressing, often linked to trauma or intense stress. "
        "It's important to find ways to cope with these experiences. "
        "Would you like to discuss grounding techniques, seeking professional help, or strategies for improving sleep?"
    ), "TRAUMA_NightmareFlashback"

def handle_normal_worry_anxiety_disorder_difference(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Understanding the difference between normal worry and an anxiety disorder is crucial for seeking the right support. "
        "Worry is a natural response to stress, while an anxiety disorder involves excessive, persistent worry that interferes with daily life. "
        "Would you like to know more about the duration and intensity of symptoms, or when to consider professional evaluation?"
    ), "INFO_NormalWorryAnxietyDisorderDifference"

def handle_ocd_intrusive_thoughts(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_ANXIOUS'])}Dealing with OCD intrusive thoughts can be incredibly distressing and feel very real, even when you know they aren't true. "
        "It's a challenging aspect of OCD. "
        "Would you like to explore strategies like Exposure and Response Prevention (ERP), mindfulness, or cognitive restructuring to manage these thoughts?"
    ), "OCD_IntrusiveThoughts"

def handle_online_therapy_effective(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Online therapy has become a very accessible and often effective way to receive mental health support. "
        "Many studies show its comparable effectiveness to in-person therapy for various conditions. "
        "Would you like to discuss its benefits, potential drawbacks, or how to choose a reputable online platform?"
    ), "RESOURCE_OnlineTherapyEffective"

def handle_overthinking_at_night(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_STRESS'])} Overthinking at night can hijack your sleep and leave you feeling exhausted. "
        "It's a common struggle when your mind won't shut off. "
        "Would you like to explore relaxation techniques before bed, journaling to clear your mind, or setting a 'worry time' during the day?"
    ), "COPING_OverthinkingAtNight"

def handle_overwhelming_stress(user_input):
    # Modified to be a short, empathetic response
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_STRESS'])} I hear you. What's been making you feel so stressed?"
    ), "EMOTIONAL_OverwhelmingStress"

def handle_panic_attack(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_ANXIOUS'])}Panic attacks are incredibly frightening, and it takes immense courage to reach out when you're experiencing one or fearing one. "
        "I'm here with you. "
        "Would you like to try a quick breathing exercise, a grounding technique, or talk about what you're feeling right now?"
    ), "COPING_PANIC" # Initiates multi-turn exercise

def handle_postpartum_depression(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_SAD'])}Postpartum depression (PPD) is a serious and common condition that can affect new mothers, and sometimes new fathers too. "
        "It's not your fault, and help is available. "
        "Would you like to learn about its symptoms, how it differs from the 'baby blues', or where to find professional support?"
    ), "INFO_PostpartumDepression"

def handle_quick_calm_anxiety_attack(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_ANXIOUS'])}When an anxiety attack hits, finding a quick way to calm down is essential. "
        "There are several techniques that can help bring you back to the present. "
        "Would you like to try a rapid breathing exercise, a sensory grounding technique, or a quick distraction method?"
    ), "COPING_QuickCalmAnxietyAttack"

def handle_quick_mindfulness_exercises(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Quick mindfulness exercises are great for bringing you into the present moment and reducing stress. "
        "They can be done almost anywhere. "
        "Would you like to try a 3-minute breathing space, a body scan, or a mindful observation exercise?"
    ), "COPING_QuickMindfulnessExercises"

def handle_reduce_social_anxiety(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_ANXIOUS'])}Reducing social anxiety can significantly improve your quality of life, allowing you to connect more freely with others. "
        "It's a gradual process, but very achievable. "
        "Would you like to discuss challenging negative thoughts, practicing social exposure, or developing coping strategies for social situations?"
    ), "ANXIETY_ReduceSocialAnxiety"

def handle_self_care_tips_for_depression(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}When dealing with depression, self-care can feel incredibly difficult, but even small steps can make a difference. "
        "It's about nurturing yourself with compassion. "
        "Would you like tips on gentle activities, maintaining routine, setting small goals, or connecting with support?"
    ), "COPING_SelfCareDepression"

def handle_set_boundaries(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Setting boundaries is a vital skill for protecting your energy, time, and emotional well-being in all relationships. "
        "It's about communicating your limits clearly. "
        "Would you like to discuss how to identify your boundaries, strategies for communicating them assertively, or dealing with people who resist them?"
    ), "WELLBEING_SetBoundaries"

def handle_signs_of_burnout(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Recognizing the signs of burnout is the first step towards recovery and preventing future episodes. "
        "It's more than just being tired; it's a state of emotional, physical, and mental exhaustion. "
        "Would you like to learn about key symptoms like chronic fatigue, cynicism, or reduced performance, and how to address them?"
    ), "INFO_SignsOfBurnout"

def handle_signs_of_poor_mental_health(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}It's important to be aware of the signs of poor mental health, both in yourself and in others, so you can seek or offer help early. "
        "These signs can be emotional, behavioral, or physical. "
        "Would you like to learn about common indicators like persistent sadness, changes in sleep/appetite, withdrawal, or increased irritability?"
    ), "INFO_SignsOfPoorMentalHealth"

def handle_stop_feeling_anxious_future(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_ANXIOUS'])}Worrying about the future is a common human experience, but when it becomes overwhelming, it can be debilitating. "
        "Let's find ways to manage those concerns. "
        "Would you like to explore mindfulness, setting realistic goals, or challenging catastrophic thinking?"
    ), "ANXIETY_FutureAnxiety"

def handle_stay_present_stop_worrying(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Learning to stay present and stop worrying can significantly reduce anxiety and improve your focus. "
        "It involves redirecting your attention to the here and now. "
        "Would you like to explore mindfulness techniques, challenging worrisome thoughts, or practical ways to anchor yourself in the present?"
    ), "COPING_StayPresentStopWorrying"

def handle_stop_comparing_myself(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_SAD'])}Constantly comparing yourself to others can be incredibly draining and harmful to your self-esteem. "
        "It's a common trap in our hyper-connected world. "
        "Would you like to discuss strategies for focusing on your own journey, practicing self-compassion, or reducing social media use?"
    ), "WELLBEING_StopComparingMyself"

def handle_stop_feeling_numb(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_SAD'])}Feeling numb can be a challenging and unsettling experience, often a way our minds cope with overwhelming emotions or trauma. "
        "It's a sign that something needs attention. "
        "Would you like to explore gentle ways to reconnect with your emotions, grounding techniques, or the importance of professional support?"
    ), "EMOTIONAL_StopFeelingNumb"

def handle_stop_negative_thoughts(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Stopping negative thoughts can feel like an uphill battle, but it's a skill you can develop. "
        "It involves recognizing, challenging, and reframing unhelpful thought patterns. "
        "Would you like to explore cognitive restructuring, thought journaling, or distraction techniques?"
    ), "COPING_StopNegativeThoughts"

def handle_stop_overthinking(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_ANXIOUS'])}Overthinking can trap you in a cycle of worry and analysis paralysis. "
        "It's a common response to stress or uncertainty. "
        "Would you like to discuss mindfulness techniques to stay present, setting limits on thought, or action-oriented strategies?"
    ), "COPING_StopOverthinking"

def handle_stop_seeking_validation(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Breaking the cycle of seeking external validation is a powerful step towards building inner confidence and self-worth. "
        "It's about shifting your focus inward. "
        "Would you like to explore ways to identify your own values, build self-compassion, or celebrate your own successes?"
    ), "WELLBEING_StopSeekingValidation"

def handle_stress_anxiety_depression_difference(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}It's common to confuse stress, anxiety, and depression, as they often overlap and can share symptoms. "
        "However, they have distinct characteristics. "
        "Would you like to learn about their key differences in triggers, duration, and impact on daily life?"
    ), "INFO_StressAnxietyDepressionDifference"

def handle_talk_about_mental_health(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Talking about mental health is a brave and important step, whether it's for yourself or to support someone else. "
        "Open communication reduces stigma and fosters understanding. "
        "Would you like tips on how to start conversations, what to share, or how to respond when someone shares with you?"
    ), "WELLBEING_TalkAboutMentalHealth"

def handle_talk_about_trauma(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['EMPATHY_SAD'])}Talking about trauma can be incredibly difficult, but it's often a crucial part of the healing process when you feel ready and safe to do so. "
        "It requires a supportive environment. "
        "Would you like to discuss why it's important to talk about trauma, how to find a safe space to share, or techniques for managing overwhelming emotions during recall?"
    ), "TRAUMA_TalkAboutTrauma"

def handle_tell_someone_about_depression(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Deciding to tell someone about your depression is a courageous and significant step towards getting support. "
        "It can feel daunting, but you don't have to carry this alone. "
        "Would you like tips on how to choose who to tell, what to say, or how to prepare for their reaction?"
    ), "RESOURCE_TellSomeoneAboutDepression"

def handle_types_of_therapy(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}There are many different types of therapy, each with its own approach and focus. "
        "Finding the right one for you depends on your needs and preferences. "
        "Would you like to learn about common types like CBT (Cognitive Behavioral Therapy), DBT (Dialectical Behavior Therapy), psychodynamic therapy, or others?"
    ), "INFO_TypesOfTherapy"

def handle_what_is_mental_health(user_input):
    return (
        f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}Mental health is about much more than the absence of mental illness; it's about your overall psychological, emotional, and social well-being. "
        "**It's incredibly important because it affects how you think, feel, and behave in daily life, impacting your relationships, work, and overall happiness.** "
        "Good mental health allows you to cope with life's stresses, realize your abilities, learn well, work productively, and contribute to your community. "
        "Would you like to explore the different components of mental health, or common factors that influence it?"
    ), "INFO_WhatIsMentalHealth"

def handle_general_positive_mood(user_input):
    return (
        random.choice(POSITIVE_RESPONSES["POSITIVE_GENERIC"])
    ), "EMOTIONAL_PositiveMood"

def handle_boredom(user_input):
    # Instead of just offering, directly give a joke or exercise sometimes
    choice = random.choice(["joke", "exercise", "offer"])
    if choice == "joke":
        return random.choice(POSITIVE_JOKES), "GENERAL_Joke"
    elif choice == "exercise":
        return handle_give_me_an_exercise(user_input) # Reuse existing exercise handler
    else: # "offer"
        return random.choice(BOREDOM_RESPONSES_OFFER), "EMOTIONAL_Boredom"


def handle_tell_me_a_joke(user_input):
    return random.choice(POSITIVE_JOKES), "GENERAL_Joke"

def handle_give_me_an_exercise(user_input):
    exercises = [
        "Let's try a quick 3-minute breathing space exercise. Find a comfortable position. Close your eyes if you feel comfortable. For the first minute, become aware of what you are experiencing. Note thoughts, feelings, and bodily sensations. For the second minute, narrow your awareness to the sensations of the breath. Follow the breath as it enters and leaves your body. For the third minute, expand your awareness back to the rest of your body and the space around you. Notice how your body feels and the sounds around you. How do you feel after that?",
        "How about a simple gratitude practice? Think of three things you are grateful for right now, big or small. What comes to mind?",
        "Let's try a quick progressive muscle relaxation. Tense the muscles in your toes tightly for 5 seconds, then release them completely. Notice the difference. Now do the same for your calves, then thighs, and so on, working your way up your body. Would you like to start with your toes?",
        "Try this physical stretch: Stand up, reach your arms overhead, stretching towards the sky as much as you can. Hold for 10 seconds, feeling your spine lengthen. Then slowly lower your arms. Repeat 3 times. How does that feel?",
        "A quick mental exercise: Think of five different colors you can see around you right now. Now, name three things that make you smile. This helps shift your focus."
    ]
    return random.choice(exercises), "EXERCISE_GENERAL"

# --- CRISIS SPECIFIC HANDLERS ---
# These return directly formatted crisis messages
def handle_crisis_suicidal(user_input):
    return CRISIS_MESSAGES["SUICIDAL_THOUGHTS"], "CRISIS_SUICIDAL" # Modified to match key

def handle_crisis_self_harm(user_input):
    return CRISIS_MESSAGES["SELF_HARM"], "CRISIS_SELF_HARM"

def handle_crisis_emergency_resources(user_input):
    return CRISIS_MESSAGES["EMERGENCY_RESOURCES"], "CRISIS_EMERGENCY_RESOURCES"

def handle_crisis_help_friend_suicidal(user_input):
    return CRISIS_MESSAGES["HELP_FRIEND_SUICIDAL"], "CRISIS_HELP_FRIEND_SUICIDAL"

def handle_crisis_psychotic_episode(user_input):
    return CRISIS_MESSAGES["PSYCHOTIC_EPISODE"], "CRISIS_PSYCHOTIC_EPISODE"

def handle_crisis_severe_distress(user_input):
    return CRISIS_MESSAGES["SEVERE_DISTRESS"], "CRISIS_SEVERE_DISTRESS"

# Helper function to check if any of the top predicted intents are crisis-related
def _check_for_crisis_intents(classification_results):
    crisis_labels = [
        "suicidal thoughts", "self harm", "emergency resources",
        "help friend suicidal", "psychotic episode", "severe distress"
    ]
    # Check top N labels to see if any crisis intent is present with a high enough score
    top_n = min(3, len(classification_results['labels'])) # Check top 3, or fewer if less are available
    crisis_threshold = 0.5 # <---- IMPORTANT: You can adjust this threshold. Lower = more sensitive to crisis.

    print(f"\n--- Debug: Checking for Crisis Intents for: {classification_results['sequence']} ---")
    for i in range(top_n):
        label = classification_results['labels'][i]
        score = classification_results['scores'][i]
        print(f"   Candidate: '{label}', Score: {score:.4f}")
        if label in crisis_labels and score >= crisis_threshold:
            print(f"   --> CRISIS DETECTED: '{label}' with score {score:.4f}")
            return label # Return the detected crisis label

    print("--- Debug: No crisis intent detected above threshold. ---")
    return None

# --- Main Structured Response Function ---
def get_structured_response(user_input):
    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session_data_updates = {} # Initialize session_data_updates here

    # This dictionary maps a classified intent tag to its corresponding handler function
    # Make sure these keys match the INTENT_LABELS exactly or are handled by fuzzy matching.
    intent_handlers = {
        "suicidal thoughts": handle_crisis_suicidal,
        "self harm": handle_crisis_self_harm,
        "emergency resources": handle_crisis_emergency_resources,
        "help friend suicidal": handle_crisis_help_friend_suicidal,
        "psychotic episode": handle_crisis_psychotic_episode,
        "severe distress": handle_crisis_severe_distress, # This is the general crisis fallback.
        "overwhelming stress": handle_overwhelming_stress, # MODIFIED
        "depressive episode": handle_depressive_episode, # MODIFIED
        "loneliness": handle_loneliness,
        "panic attack": handle_panic_attack,
        "health anxiety": handle_health_anxiety,
        "imposter syndrome": handle_imposter_syndrome,
        "breakup grief": handle_breakup_grief,
        "drained after socializing": handle_drained_after_socializing,
        "ocd intrusive thoughts": handle_ocd_intrusive_thoughts,
        "manic behavior": handle_manic_behavior,
        "cannot sleep tired": handle_cant_sleep_tired,
        "feeling burned out": handle_burned_out,
        "good sleep habits": handle_good_sleep_habits,
        "manage work related stress": handle_manage_work_related_stress,
        "motivate when worthless": handle_motivate_worthless,
        "overthinking at night": handle_overthinking_at_night,
        "signs of burnout": handle_signs_of_burnout,
        "balance life avoid stress": handle_balance_life_stress,
        "nightmare flashback": handle_nightmare_flashback,
        "feeling sad all the time": handle_feeling_sad_all_the_time, # MODIFIED
        "anger management": handle_anger_management,
        "know if anxiety": handle_know_if_anxiety,
        "reduce social anxiety": handle_reduce_social_anxiety,
        "stop feeling anxious about the future": handle_stop_feeling_anxious_future,
        "am i depressed or sad": handle_depressed_or_sad_diff, # Will route here, but handler has internal logic
        "self-care tips for depression": handle_self_care_tips_for_depression,
        "stop feeling numb": handle_stop_feeling_numb,
        "cry for no reason": handle_cry_for_no_reason,
        "tell someone about depression": handle_tell_someone_about_depression,
        "set boundaries": handle_set_boundaries,
        "quick mindfulness exercises": handle_quick_mindfulness_exercises,
        "meditation helps mental health": handle_meditation_help,
        "breathing techniques anxiety": handle_breathing_anxiety,
        "stay present stop worrying": handle_stay_present_stop_worrying,
        "adhd overwhelm": handle_adhd_overwhelm,
        "bad day calm down": handle_bad_day_calm_down, # MODIFIED
        "build emotional resilience": handle_build_emotional_resilience,
        "build self confidence": handle_build_self_confidence,
        "cannot afford therapy": handle_cant_afford_therapy,
        "deal with guilt or shame": handle_deal_with_guilt_shame,
        "deal with toxic person": handle_deal_with_toxic_person,
        "elderly isolation": handle_elderly_isolation,
        "emotional triggers": handle_emotional_triggers,
        "feeling not good enough": handle_not_good_enough,
        "find good therapist": handle_find_good_therapist,
        "grounding techniques flashbacks": handle_grounding_flashbacks,
        "having nightmares": handle_having_nightmares,
        "heal from past trauma": handle_heal_trauma,
        "healthy coping stress": handle_healthy_coping_stress,
        "help partner struggling": handle_help_partner_struggling,
        "improve communication in relationships": handle_improve_communication_relationships,
        "improve mental well-being": handle_improve_mental_wellbeing,
        "know if need therapy": handle_know_if_anxiety,
        "know if ptsd": handle_know_if_ptsd,
        "make friends social anxiety": handle_make_friends_social_anxiety,
        "manage overwhelming emotions": handle_manage_overwhelming_emotions,
        "mental health physical health link": handle_mental_health_physical_health_link,
        "narcissistic parent": handle_narcissistic_parent,
        "need mental health support": handle_need_mental_health_support,
        "normal worry anxiety disorder difference": handle_normal_worry_anxiety_disorder_difference,
        "online therapy effective": handle_online_therapy_effective,
        "postpartum depression": handle_postpartum_depression,
        "quick calm anxiety attack": handle_quick_calm_anxiety_attack,
        "signs of poor mental health": handle_signs_of_poor_mental_health,
        "stop comparing myself": handle_stop_comparing_myself,
        "stop negative thoughts": handle_stop_negative_thoughts,
        "stop overthinking": handle_stop_overthinking,
        "stop seeking validation": handle_stop_seeking_validation,
        "stress anxiety depression difference": handle_stress_anxiety_depression_difference,
        "talk about mental health": handle_talk_about_mental_health,
        "talk about trauma": handle_talk_about_trauma,
        "types of therapy": handle_types_of_therapy,
        "what is mental health": handle_what_is_mental_health,
        "boredom": handle_boredom,
        "general positive mood": handle_general_positive_mood, # Directly map to the handler
        "tell me a joke": handle_tell_me_a_joke, # New handler for joke requests
        "give me an exercise": handle_give_me_an_exercise # New handler for exercise requests
    }

    user_input_lower = user_input.lower()

    # 1. Offensive Language Check (Highest Priority)
    if any(keyword in user_input_lower for keyword in OFFENSIVE_KEYWORDS):
        response_text = random.choice(OFFENSIVE_RESPONSES)
        intent_tag = "OFFENSIVE_LANGUAGE"
        log_interaction(current_timestamp, user_input, response_text, intent_tag, detected_intent="Offensive Language Detected")
        return response_text, intent_tag, session_data_updates

    # 2. Hardcoded specific checks for common, simple phrases before classifier
    # This ensures common greetings and jokes are handled quickly and accurately
    if user_input_lower in ["hi", "hello", "hey", "hallo", "good morning", "good afternoon", "good evening"]:
        response_text = random.choice(GREETINGS)
        intent_tag = "GENERAL_GREETING"
        log_interaction(current_timestamp, user_input, response_text, intent_tag, detected_intent="Greeting")
        return response_text, intent_tag, session_data_updates
    elif "joke" in user_input_lower: # Keep this as a general fallback for joke requests
        response_text = random.choice(POSITIVE_JOKES)
        intent_tag = "GENERAL_Joke"
        log_interaction(current_timestamp, user_input, response_text, intent_tag, detected_intent="Joke Request")
        return response_text, intent_tag, session_data_updates
    elif "i'm bored" in user_input_lower or "i am bored" in user_input_lower or user_input_lower == "bored":
        # Direct call to handle_boredom for precise control over boredom responses
        response_text, intent_tag = handle_boredom(user_input)
        log_interaction(current_timestamp, user_input, response_text, intent_tag, detected_intent="Boredom (Direct Match)")
        return response_text, intent_tag, session_data_updates
    elif "exercise" in user_input_lower or "activity" in user_input_lower:
        response_text, intent_tag = handle_give_me_an_exercise(user_input)
        log_interaction(current_timestamp, user_input, response_text, intent_tag, detected_intent="Exercise Request (Direct Match)")
        return response_text, intent_tag, session_data_updates


    # If the classifier failed to load, return a graceful fallback
    if classifier is None:
        print("Classifier is not loaded, returning default error.")
        response_text = "I'm currently experiencing technical difficulties with understanding complex requests. Please try again later, or contact support if the issue persists."
        intent_tag = "ERROR_CLASSIFIER_UNAVAILABLE"
        log_interaction(current_timestamp, user_input, response_text, intent_tag, detected_intent="Classifier Unavailable")
        return response_text, intent_tag, session_data_updates


    try:
        # First, classify against all possible intents to get a full picture
        classification_results = classifier(user_input, INTENT_LABELS, multi_label=True)
        # Assuming the first result is the most relevant one for general classification
        top_prediction = classification_results['labels'][0]
        top_score = classification_results['scores'][0]

        print(f"\n--- Debug: User input: '{user_input}' ---")
        print(f"--- Debug: Top general classified intent: '{top_prediction}' with score: {top_score:.4f} ---")
        print(f"--- Debug: Full classification results: {classification_results} ---")


        # --- IMPORTANT: CRISIS INTENT OVERRIDE ---
        # Check if any crisis intent is present with a reasonable score among the top predictions.
        # Pass the full classification results to the helper function.
        detected_crisis_intent = _check_for_crisis_intents(classification_results)
        if detected_crisis_intent:
            # Crisis message keys in CRISIS_MESSAGES should match the INTENT_LABELS, but with underscores and uppercase
            # E.g., "suicidal thoughts" -> "SUICIDAL_THOUGHTS"
            crisis_message_key = detected_crisis_intent.replace(" ", "_").upper()
            crisis_response = CRISIS_MESSAGES.get(crisis_message_key)

            if crisis_response: # Ensure the message exists in CRISIS_MESSAGES
                log_interaction(current_timestamp, user_input, crisis_response, f"CRISIS_{crisis_message_key}", detected_intent=detected_crisis_intent, classifier_score=top_score)
                return crisis_response, f"CRISIS_{crisis_message_key}", session_data_updates
            else:
                # Fallback if crisis intent detected but no specific message defined (shouldn't happen if constants are correct)
                response_text = CRISIS_MESSAGES["SEVERE_DISTRESS"] # Default to severe distress if specific message is missing
                intent_tag = f"CRISIS_FALLBACK_{crisis_message_key}"
                log_interaction(current_timestamp, user_input, response_text, intent_tag, detected_intent=detected_crisis_intent, classifier_score=top_score)
                return response_text, intent_tag, session_data_updates


        # You might want a confidence threshold here for general intents
        CONFIDENCE_THRESHOLD = 0.7 # Adjust as needed. Lower = more general responses, Higher = more specific.

        if top_score >= CONFIDENCE_THRESHOLD:
            # All crisis intents are handled by the _check_for_crisis_intents above.
            # Now handle other high-confidence intents.

            if top_prediction in intent_handlers:
                response_text, intent_tag = intent_handlers[top_prediction](user_input)
                # If an exercise is being *initiated*, set the session_data_updates
                if intent_tag in ["COPING_PANIC", "EXERCISE_ANGER_STEP1", "COPING_ADHD", "COPING_QuickCalmAnxietyAttack"]:
                    # Note: For COPING_QuickCalmAnxietyAttack, the handler itself might set the initial step.
                    # Ensure the handler returns the correct initial step, or set it here.
                    # For simplicity, let's assume handlers initiating exercises set their own step 1.
                    # So, we just need to pass the type.
                    session_data_updates['exercise_state'] = {'type': intent_tag, 'current_step': 'step1'} # Assume step1 is always the start
                    print(f"--- Debug: Initiating exercise: {intent_tag} ---")

                log_interaction(current_timestamp, user_input, response_text, intent_tag, detected_intent=top_prediction, classifier_score=top_score)
                return response_text, intent_tag, session_data_updates
            else:
                # Fallback if a classified intent doesn't have a handler
                print(f"Warning: Classified intent '{top_prediction}' has no specific handler. Falling back to general.")
                response_text = f"{random.choice(EMOTIONAL_INTERJECTIONS['ACKNOWLEDGEMENT'])}I understand you're interested in '{top_prediction}'. Can you tell me more about what you'd like to know or discuss regarding this?"
                intent_tag = "DEVELOPMENT_ERROR_MISSING_HANDLER" # This indicates a handler function should be created
                log_interaction(current_timestamp, user_input, response_text, intent_tag, detected_intent=top_prediction, classifier_score=top_score)
                return response_text, intent_tag, session_data_updates
        else:
            # If confidence is below threshold for ALL non-crisis intents, fall back to generative or general.
            print(f"Classifier confidence too low ({top_score:.2f}) for '{top_prediction}'. Falling back to general or generative model.")
            # The chatbot.py's get_haven_response already handles the DialoGPT fallback if get_structured_response returns None
            return None, "NO_STRUCTURED_RESPONSE", session_data_updates # Return None to trigger DialoGPT in chatbot.py

    except Exception as e:
        print(f"Error during zero-shot classification: {e}")
        traceback.print_exc() # Print the full traceback
        # Always return an explicit error message here if classification itself fails
        response_text = "I'm experiencing a technical issue with understanding. Could you please try again or rephrase?"
        intent_tag = "CLASSIFICATION_ERROR"
        log_interaction(current_timestamp, user_input, response_text, intent_tag, detected_intent="CLASSIFICATION_ERROR", classifier_score=0)
        return response_text, intent_tag, session_data_updates


# --- Handle multi-turn exercise logic ---
# This function manages the state of an ongoing exercise
# It should return (response_text, is_exercise_complete)
EXERCISE_STEPS = {
    "EXERCISE_ANGER_STEP1": { # Renamed to match the intent tag that initiates it
        "step1": {
            "prompt": "The 'STOP' method is a quick way to gain control. It stands for Stop, Take a breath, Observe, and Proceed. Would you like to try it? (Yes/No)",
            "next_step": "step2_stop"
        },
        "step2_stop": {
            "prompt": "Okay, let's try 'STOP'. First, **Stop** what you're doing. Physically pause. Take a deep, slow **breath**. Now, **Observe** your thoughts and feelings without judgment. What are you noticing?",
            "next_step": "step3_proceed",
            "complete_message": "That's a great start to observing your anger. Remember, you can use 'STOP' anytime you feel anger rising."
        },
        "step3_proceed": {
            "prompt": "Excellent. Now, **Proceed** mindfully. Based on what you observed, what's one small, constructive step you can take right now, or what do you need?",
            "complete_message": "Great job. Remember, proceeding mindfully means choosing your response rather than reacting. You can always come back to this when you need it.",
            "final": True # Marks this as the final step
        }
    },
    "COPING_PANIC": { # Matches the intent tag
        "step1": {
            "prompt": "Let's try a quick 4-7-8 breathing exercise. It's simple and effective. Are you ready to start? (Yes/No)",
            "next_step": "step2_breathe"
        },
        "step2_breathe": {
            "prompt": "Okay, great. Breathe in quietly through your nose for a count of **4**. Hold your breath for a count of **7**. Then, exhale completely through your mouth, making a 'whoosh' sound, for a count of **8**. Let's do one cycle. Ready to try another or tell me how you feel?",
            "next_step": "step3_reflect",
            "complete_message": "That's one cycle of 4-7-8 breathing. You can repeat this several times until you feel calmer."
        },
        "step3_reflect": {
            "prompt": "How are you feeling after that cycle? Would you like to do another, or would you prefer a different calming technique?",
            "complete_message": "Remember, breathing exercises are a powerful tool you can use anytime to help regain calm.",
            "final": True
        }
    },
    "COPING_ADHD": { # Matches the intent tag
        "step1": {
            "prompt": "I can help with task breakdown or distraction management for ADHD. Which would you prefer? (Breakdown/Distractions)",
            "next_step_breakdown": "step2_breakdown_task",
            "next_step_distractions": "step3_distraction_choice"
        },
        "step2_breakdown_task": {
            "prompt": "Breaking down tasks into smaller, manageable steps is very effective. Let's take one task you're struggling with. Can you tell me what it is?",
            "complete_message": "Okay, that task can be broken down. Remember, starting small helps overcome overwhelm. We can break down more tasks anytime.",
            "final": True
        },
        "step3_distraction_choice": {
            "prompt": "Managing distractions is key for ADHD. Would you like to set up a 'focus environment' or try a 'Pomodoro Technique' session?",
            "complete_message": "Great, let's try that. Consistency is key for managing distractions. We can explore more techniques anytime.",
            "final": True
        }
    },
    "COPING_QuickCalmAnxietyAttack": { # Matches the intent tag
        "step1": {
            "prompt": "Let's try a quick sensory grounding technique. Can you name 5 things you can see around you right now?",
            "next_step": "step2_hear"
        },
        "step2_hear": {
            "prompt": "Excellent. Now, can you name 4 things you can hear?",
            "next_step": "step3_touch"
        },
        "step3_touch": {
            "prompt": "Perfect. Now, 3 things you can touch?",
            "next_step": "step4_smell"
        },
        "step4_smell": {
            "prompt": "Almost there! 2 things you can smell?",
            "next_step": "step5_taste"
        },
        "step5_taste": {
            "prompt": "And finally, 1 thing you can taste? Once you've gone through these senses, take a deep breath. This helps bring you back to the present. Are you feeling more grounded now?",
            "complete_message": "Great job with the grounding exercise. Remember, you can use this technique anytime you feel overwhelmed to bring yourself back to the present moment.",
            "final": True
        }
    }
}


def handle_ongoing_exercise(user_input, exercise_state):
    exercise_type = exercise_state.get('type')
    current_step_key = exercise_state.get('current_step')

    print(f"--- Debug: handle_ongoing_exercise called. Type: {exercise_type}, Current Step: {current_step_key}, User Input: '{user_input}' ---")

    if not exercise_type or not current_step_key or exercise_type not in EXERCISE_STEPS:
        print("--- Debug: Invalid exercise state. Resetting. ---")
        return "I'm sorry, I seem to have lost track of our exercise. Can we start a new one?", True

    current_exercise_definition = EXERCISE_STEPS[exercise_type]
    current_step_details = current_exercise_definition.get(current_step_key)

    if not current_step_details:
        print(f"--- Debug: Current step '{current_step_key}' not found in exercise '{exercise_type}'. Resetting. ---")
        return "It seems there was an issue with the exercise step. Let's try again.", True

    user_input_lower = user_input.lower().strip()
    response_text = None
    exercise_complete = False

    # Check for explicit completion signals first
    if "done" in user_input_lower or "finished" in user_input_lower or "stop" in user_input_lower or current_step_details.get("final"):
        response_text = current_step_details.get("complete_message", "Exercise complete. How are you feeling now?")
        exercise_complete = True
        print(f"--- Debug: Exercise '{exercise_type}' explicitly completed by user or final step. ---")
        return response_text, exercise_complete

    # Specific logic for each exercise type and step
    if exercise_type == "EXERCISE_ANGER_STEP1":
        if current_step_key == "step1":
            if "yes" in user_input_lower or "try" in user_input_lower:
                exercise_state['current_step'] = "step2_stop"
                response_text = current_exercise_definition["step2_stop"]["prompt"]
            elif "no" in user_input_lower or "not now" in user_input_lower:
                response_text = "No problem. Would you prefer to just talk about what's making you angry instead?"
                exercise_complete = True
            else:
                response_text = current_step_details["prompt"] # Repeat prompt if unclear
        elif current_step_key == "step2_stop":
            # User has observed, now move to proceed
            exercise_state['current_step'] = "step3_proceed"
            response_text = current_exercise_definition["step3_proceed"]["prompt"]
        elif current_step_key == "step3_proceed":
            # This is the final step, any input completes it
            response_text = current_step_details["complete_message"]
            exercise_complete = True

    elif exercise_type == "COPING_PANIC":
        if current_step_key == "step1":
            if "yes" in user_input_lower or "breathing" in user_input_lower:
                exercise_state['current_step'] = "step2_breathe"
                response_text = current_exercise_definition["step2_breathe"]["prompt"]
            elif "no" in user_input_lower or "grounding" in user_input_lower:
                # If user says no to breathing, suggest grounding from QuickCalmAnxietyAttack
                exercise_state['type'] = "COPING_QuickCalmAnxietyAttack" # Change exercise type
                exercise_state['current_step'] = "step1" # Start grounding from its step1
                response_text = EXERCISE_STEPS["COPING_QuickCalmAnxietyAttack"]["step1"]["prompt"]
            else:
                response_text = current_step_details["prompt"] # Repeat prompt
        elif current_step_key == "step2_breathe":
            # User has tried breathing, now ask how they feel/if they want more
            exercise_state['current_step'] = "step3_reflect"
            response_text = current_exercise_definition["step3_reflect"]["prompt"]
        elif current_step_key == "step3_reflect":
            # This is the final step, any input completes it
            response_text = current_step_details["complete_message"]
            exercise_complete = True

    elif exercise_type == "COPING_ADHD":
        if current_step_key == "step1":
            if "breakdown" in user_input_lower:
                exercise_state['current_step'] = "step2_breakdown_task"
                response_text = current_exercise_definition["step2_breakdown_task"]["prompt"]
            elif "distractions" in user_input_lower:
                exercise_state['current_step'] = "step3_distraction_choice"
                response_text = current_exercise_definition["step3_distraction_choice"]["prompt"]
            else:
                response_text = current_step_details["prompt"] # Repeat prompt
        elif current_step_key == "step2_breakdown_task":
            response_text = current_step_details["complete_message"]
            exercise_complete = True
        elif current_step_key == "step3_distraction_choice":
            response_text = current_step_details["complete_message"]
            exercise_complete = True

    elif exercise_type == "COPING_QuickCalmAnxietyAttack": # This is the grounding exercise
        if current_step_key == "step1": # 5 things you see
            exercise_state['current_step'] = "step2_hear"
            response_text = current_exercise_definition["step2_hear"]["prompt"]
        elif current_step_key == "step2_hear": # 4 things you hear
            exercise_state['current_step'] = "step3_touch"
            response_text = current_exercise_definition["step3_touch"]["prompt"]
        elif current_step_key == "step3_touch": # 3 things you touch
            exercise_state['current_step'] = "step4_smell"
            response_text = current_exercise_definition["step4_smell"]["prompt"]
        elif current_step_key == "step4_smell": # 2 things you smell
            exercise_state['current_step'] = "step5_taste"
            response_text = current_exercise_definition["step5_taste"]["prompt"]
        elif current_step_key == "step5_taste": # 1 thing you taste (final)
            response_text = current_step_details["complete_message"]
            exercise_complete = True

    # Default fallback if no specific logic matched for the current step, but exercise is still active
    if response_text is None:
        response_text = current_step_details.get("prompt", "I'm not sure how to proceed with this exercise. Can we restart or try something else?")
        # If no specific next step logic, and not explicitly completed, assume it's still ongoing unless final.
        exercise_complete = current_step_details.get("final", False)

    print(f"--- Debug: handle_ongoing_exercise returning: '{response_text}', Complete: {exercise_complete} ---")
    return response_text, exercise_complete