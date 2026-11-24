SYSTEM_PROMPT = 'You are a helpful assistant that carefully analyzes audio.'


QA_PROMPT = (
    'Listen to the audio and answer the following question.\n'
    '\n'
    'Question: {question}\n'
    '{options}\n'
    '\n'
    'Answer:'
)


ASR_PROMPT = 'Transcribe the English audio into text.'


EVENT_PROMPT = (
    'Generate {k} sound event descriptions suitable for text-to-audio generation.'
    ' Each description should describe clearly audible sound events that fit '
    'within a 10-second clip. Cover a wide range of settings: wildlife, weather, '
    'environmental sound, machines, musical instruments, human activity, and so '
    'on.{examples}Reply with ONLY a JSON list of {k} strings.'
)


EVENT_EXAMPLES = [
    'an owl hooting twice in a forest as crickets chirp',
    'a car engine starting up and idling for three seconds',
    'a group of people laughing and chatting while a baby cries',
]


TIMED_EVENTS_PROMPT = (
    'Generate {k} sound scenes for text-to-audio generation. Describe each scene '
    "as two to four distinct sound events separated by ' ; ', in the order they "
    'should be heard. Keep every event a short lowercase sound description of at '
    'most eight words. Do NOT write any numbers, timings, durations or counts, '
    'only the sounds themselves.{examples}Reply with ONLY a JSON list of {k} '
    'strings.'
)


TIMED_EVENTS_EXAMPLES = [
    'an ambulance siren wailing ; a chainsaw cutting wood ; a subway train passing',
    'thunder rumbling ; a stream flowing over rocks',
    (
        'children playing in a yard ; a metal gate banging ; a dog barking nearby'
        ' ; wind gusting through trees'
    ),
]


FREQUENCY_EVENTS_PROMPT = (
    'Generate {k} sound scenes for text-to-audio generation. Describe each scene '
    "as one to four distinct simple-present sound-event clauses separated by ' ; "
    "'. Every clause must be a complete short phrase that can stand as a "
    'sentence. Do NOT write any numbers, timings, durations, counts, frequency '
    "words, or words such as 'repeatedly'; only describe the sounds themselves. "
    'Cover diverse settings.{examples}Reply with ONLY a JSON list of {k} strings.'
)


FREQUENCY_EVENTS_EXAMPLES = [
    'a dog barks ; footsteps approach',
    'a hammer strikes metal ; an alarm rings ; a door slams',
    'a crow caws ; leaves rustle ; a bicycle bell rings',
]


AUDIO_CATEGORIES = {
    'duration': {
        'prompt': TIMED_EVENTS_PROMPT,
        'examples': TIMED_EVENTS_EXAMPLES,
        'timed': 'durations',
    },
    'frequency': {
        'prompt': FREQUENCY_EVENTS_PROMPT,
        'examples': FREQUENCY_EVENTS_EXAMPLES,
        'counted': True,
    },
    'ordering': {
        'prompt': TIMED_EVENTS_PROMPT,
        'examples': TIMED_EVENTS_EXAMPLES,
        'timed': 'sequence',
    },
    'timed_composition': {
        'prompt': TIMED_EVENTS_PROMPT,
        'examples': TIMED_EVENTS_EXAMPLES,
        'timed': 'overlap',
    },
    'overlap_simultaneity': {
        'prompt': (
            'Generate {k} sound scene descriptions for text-to-audio generation. '
            'Each scene should contain two sounds that overlap in time. Make it '
            'clear that the two sounds happen at the same time, such as a longer '
            'sound continuing while a shorter sound occurs over '
            'it.{examples}Reply with ONLY a JSON list of {k} strings.'
        ),
        'examples': [
            'Rain falls steadily on a roof while a dog barks intermittently over it.',
            (
                'A kettle whistles continuously as footsteps cross a wooden floor'
                ' at the same time.'
            ),
            'Wind howls throughout the clip while a metal gate bangs repeatedly.',
        ],
    },
    'negation_absence': {
        'prompt': (
            'Generate {k} sound scene descriptions for text-to-audio generation. '
            'Each description should include one or two sounds that are present '
            'and exactly one related sound that is absent. State the absent sound'
            " using 'no', 'without', or 'never'. Choose a sound that would "
            'normally be expected to occur with the other sounds.{examples}Reply '
            'with ONLY a JSON list of {k} strings.'
        ),
        'examples': [
            'Heavy rain drums on a roof, but there is no thunder at any point.',
            (
                'A busy street with engines rumbling and tires rolling, without a'
                ' single car horn.'
            ),
            'A campfire crackles steadily, with no wind noise at all.',
        ],
    },
    'multi_event_count': {
        'prompt': (
            'Generate {k} sound scene descriptions for text-to-audio generation. '
            'Each scene should include three to five different sound sources, and'
            ' all of them should be clearly audible in the clip. You may state '
            'the exact number of sources when useful. Keep the sounds distinct '
            'enough to tell them apart.{examples}Reply with ONLY a JSON list of '
            '{k} strings.'
        ),
        'examples': [
            'heavy rain pours, a thunderclap booms, wind howls, and a window rattles.',
            (
                'Exactly three different sounds, one after another: a dog '
                'barking, a phone ringing, and glass shattering.'
            ),
            (
                'A kitchen where a kettle whistles, a knife chops on a cutting '
                'board, and a pan sizzles.'
            ),
        ],
    },
}


TEXT_PROMPT = (
    'Generate {k} English sentences suitable for text-to-speech generation. Each '
    'sentence should be natural and speakable within a 10-second '
    'clip.{examples}Reply with ONLY a JSON list of {k} strings.'
)


TEXT_EXAMPLES = [
    'The package arrived earlier than expected.',
    'Oh my god, I completely forgot about the meeting this morning!',
    (
        'After searching for nearly an hour, we finally found the missing keys '
        'under the sofa.'
    ),
]


SPEECH_CATEGORIES = {
    'general': {
        'prompt': TEXT_PROMPT,
        'examples': TEXT_EXAMPLES,
        'min_words': 4,
        'max_words': 22,
    },
    'lexical_ambiguity_idiom': {
        'prompt': (
            'Generate {k} English sentences for text-to-speech generation. Each '
            'sentence should contain exactly one meaningful lexical challenge: a '
            'common idiom, a polysemous word used in one specific sense, a '
            'homophone whose intended spelling is clear from context, or a short '
            'wordplay expression. Provide enough natural context inside the '
            'sentence to make the intended interpretation unambiguous, but do not'
            ' explain or define the expression. Avoid obscure cultural references'
            ' and quotations. Keep each item to one self-contained sentence that '
            'can be spoken within 10 seconds.{examples}Reply with ONLY a JSON '
            'list of {k} strings.'
        ),
        'examples': [
            (
                'After weeks of delays, the manager finally decided to pull the '
                'plug on the project.'
            ),
            'The hikers rested on the river bank before continuing along the trail.',
            'The wind was too strong for the boat to sail safely across the bay.',
        ],
        'min_words': 8,
        'max_words': 20,
    },
    'contrastive_phonetic_content': {
        'prompt': (
            'Generate {k} natural English sentences for text-to-speech '
            'generation. Each sentence should contain one clearly identifiable '
            'set of two or three easily confused words that must be pronounced '
            'distinctly. Use minimal or near-minimal contrasts such as a single '
            'vowel difference, a single consonant difference, a voicing contrast,'
            ' or a confusable number pair. Include every contrasting word '
            'explicitly in the same sentence and provide enough context to make '
            'each word appropriate. Do not use exact homophones, random word '
            'lists, phonetic notation, or tongue twisters. Keep each item to one '
            'sentence that can be spoken within 10 seconds.{examples}Reply with '
            'ONLY a JSON list of {k} strings.'
        ),
        'examples': [
            'She ordered fourteen folders, not forty folders, for the meeting.',
            'The ship passed the sheep grazing near the shore.',
            'Please pack the blue cap beside the blue cab.',
        ],
        'min_words': 7,
        'max_words': 18,
    },
}


QUESTION_PROMPT = (
    'You are given an audio clip and its reference caption.\n'
    'Reference caption: {ref}\n'
    '\n'
    'Listen carefully to the audio and generate {n} diverse multiple-choice '
    'questions that assess understanding of this specific audio clip. Use the '
    'caption to identify candidate facts, but confirm them from the audio. Do not'
    ' ask about any detail that is not audibly verifiable. The trainee will '
    'receive the audio and question, but not the caption.\n'
    '\n'
    'Requirements:\n'
    '- Each question must require listening to the audio to answer correctly.\n'
    '- Cover a variety of acoustic aspects, such as sound events, temporal order,'
    ' the overall scene and ambience, speech (e.g., speaker characteristics, '
    'emotion, or spoken content), and music (e.g., instruments, tempo, rhythm, or'
    ' mood) when present.\n'
    '- Provide exactly four answer options for each question, with exactly one '
    'correct answer.\n'
    '- Incorrect options should be plausible for a similar audio clip but clearly'
    ' incorrect for this one. Good distractors often differ from the correct '
    'answer in just one detail, such as the order of two events, a count, a '
    'duration, or an attribute of a sound.\n'
    '\n'
    'Reply with ONLY a JSON list in the following format:\n'
    '[{{"question": "...", "options": ["...", "...", "...", "..."], '
    '"answer_index": 0}}]'
)


VOTE_PROMPT = (
    'Listen to the audio. First describe briefly what you hear (one or two '
    'sentences). Then answer the question.\n'
    '\n'
    'Question: {question}\n'
    '{options}\n'
    '\n'
    'End your reply with exactly: Final answer: (X)'
)


DIALOGUE_CAPABILITIES = {
    'conversational_fact_retrieval': (
        'Include explicit facts such as names, locations, quantities, plans, or '
        'preferences that can be retrieved from what the speakers say.'
    ),
    'role_and_relationship': (
        'Include spoken clues that establish roles or relationships, such as '
        'customer and employee, teacher and student, coworkers, or relatives.'
    ),
    'deixis_and_coreference': (
        'Include pronouns or referring expressions such as it, they, that one, '
        'there, or the earlier request, whose referent can be resolved from '
        'previous spoken turns.'
    ),
    'intent_and_speech_act': (
        'Include a clearly spoken request, refusal, promise, apology, warning, '
        'correction, suggestion, or confirmation whose communicative intent can '
        'be identified from the dialogue.'
    ),
    'dialogue_scene_reasoning': (
        'Include spoken clues from which the likely setting or ongoing activity '
        'can be inferred. Do not rely on background sounds or narration.'
    ),
}


DIALOGUE_PLAN_PROMPT = (
    'Write exactly {k} substantially different outlines for multi-speaker '
    'dialogues.\n'
    '\n'
    'Each dialogue should be designed to naturally exercise the target capability'
    ' specified below. Use the capability key as the task category, and use the '
    'capability definition to understand what kind of information, reasoning, or '
    'interaction the future dialogue should require.\n'
    '\n'
    'Capability key: {capability}\n'
    'Capability definition: {capability_description}\n'
    '\n'
    'At this stage, generate planning briefs only. Do not write any part of the '
    'actual conversation.\n'
    '\n'
    'Requirements:\n'
    '- Each brief should describe what the future conversation is about, the '
    'situation in which it takes place, how it develops, and what the final '
    'outcome it reaches.\n'
    '- Choose a concrete main topic and a plausible setting.\n'
    '- Specify 2 to 6 participant roles.\n'
    '- Write the summary in third-person planning prose. It should cover the '
    'premise, the important information exchanged, the progression of the '
    'situation, and the final outcome.\n'
    '- Design the situation so that demonstrating the target capability is '
    'necessary or useful for successfully completing the conversation.\n'
    '- Make all {k} briefs different from one another\n'
    '- The topic is not limited to everyday, professional, civic, educational, '
    'travel, service, hobby, or family contexts, and may extend beyond generic '
    'meetings, cafes, appointments, project deadlines, or reservations.\n'
    '- The entire situation must be expressible through spoken dialogue alone. Do'
    ' not rely on narration, visual information, or background sounds to make the'
    ' scenario understandable.\n'
    '\n'
    'Reply with ONLY a JSON list of planning briefs. Each item must contain '
    'exactly these five fields and no additional fields, especially no dialogue '
    'or script field:\n'
    '[{{"topic": "...", "setting": "...", "participants": ["...", "..."], '
    '"summary": "...", "key_details": ["...", "..."]}}]'
)


DIALOGUE_SCRIPT_PROMPT = (
    'Write exactly one English multi-speaker dialogue based on the planning '
    'context below.\n'
    'Capability key: {capability}\n'
    'Capability definition: {capability_description}\n'
    '\n'
    'Planning context (follow it closely; do not replace it with a generic '
    'scenario):\n'
    '{planning_context}\n'
    '\n'
    'Requirements:\n'
    '- Use 2 to 6 speakers and 5 to 10 turns.\n'
    '- Adjacent turns must use different speakers.\n'
    '- Every speaker must speak at least once\n'
    '- Include explicit spoken evidence needed for the requested capability.\n'
    '- Construct the dialogue so the spoken turns contain all evidence needed to '
    'test the capability definition.\n'
    '- Use natural facts, references, corrections, requests, or role clues.\n'
    '- Do not include text that should not be spoken aloud.\n'
    '\n'
    'Reply with ONLY a JSON list containing exactly one item. It must have this '
    'shape:\n'
    '[{{"scenario": "...", "speakers": ["S1", "S2"], "turns": [{{"speaker": "S1",'
    ' "text": "..."}}, {{"speaker": "S2", "text": "..."}}]}}]'
)


DIALOGUE_QUESTION_PROMPT = (
    'You are given a multi-speaker dialogue and its reference transcript.\n'
    'Reference transcript:\n'
    '{ref}\n'
    '\n'
    'Listen to the complete audio and generate {n} multiple-choice questions for '
    'the assigned capability below.\n'
    'Capability key: {capability}\n'
    'Capability definition: {capability_description}\n'
    '\n'
    'Use the transcript only to identify candidate facts. You should design the '
    'question, confirm each fact, and the correct answer from the audio.\n'
    'Requirements:\n'
    '- Every question must require listening to this dialogue.\n'
    '- Do not ask about any detail that is not audibly verifiable.\n'
    '- Do not generate speaker-count, turn-count, total-duration, timestamp, or '
    'utterance-start-time questions. - Never refer to hidden labels such as S1, '
    'S2, Speaker A, or Speaker B.\n'
    "- Refer to a voice by audible order or spoken evidence, such as 'the first "
    "speaker' or 'the person who mentioned the reservation'.\n"
    '- Give exactly four distinct plausible options and exactly one correct '
    'answer.\n'
    '\n'
    'Reply with ONLY a JSON list:\n'
    '[{{"question": "...", "options": ["...", "...", "...", "..."], '
    '"answer_index": 0}}]'
)


PERTURB_AUDIO_PROMPT = (
    'You will be given a description of a 10-second sound clip.\n'
    'Rewrite it into {k} DIFFERENT variants. Each variant should describe a clip '
    'that would not match the original description. For each variant, make one of'
    ' the following changes while keeping everything else the same: change the '
    'order of events, change a timestamp or duration, change how many times a '
    'sound occurs, switch an attribute between two sounds, change a sound that is'
    ' absent to present or a present sound to absent, add a new sound event, or '
    'remove one of the described sound events. Each change should be as small as '
    'possible, so the resulting clip remains highly similar to the original while'
    ' still sounding clearly different.Make sure the variants are also different '
    'from one another.\n'
    'Description: {instruction}\n'
    'Reply with ONLY a JSON list of {k} strings.'
)


PERTURB_SPEECH_PROMPT = (
    'You will be given a sentence that a voice will speak aloud in a 10-second '
    'clip.\n'
    'Rewrite it into {k} DIFFERENT variants that sound very similar when spoken, '
    'but are not exactly the same words. A recording of the original sentence '
    'should not match any of the variants exactly. For each variant, make one of '
    'the following changes while leaving the rest of the sentence unchanged: '
    'replace a word with a word that sounds very similar when spoken (for '
    "example, 'seventy' can become 'seventeen', 'bell' can become 'bill', or "
    "'lock' can become 'log'), or change how many times a repeated word appears. "
    'The replacement word must sound very similar to the original word when '
    'spoken. Do not use synonyms or words that sound noticeably different. Keep '
    'the length and rhythm as close to the original as possible, and make sure '
    'every variant sounds natural when spoken.\n'
    'Sentence: {instruction}\n'
    'Reply with ONLY a JSON list of {k} strings.'
)


PERTURB_AUDIO_SHOTS = [
    (
        'a dog barking three times, then a door slamming',
        [
            'a dog barking five times, then a door slamming',
            'a door slamming, then a dog barking three times',
            'a dog barking three times, with no door slamming at any point',
        ],
    ),
    (
        (
            'rain falling steadily for six seconds, then a thunderclap at the '
            'eight-second mark'
        ),
        [
            (
                'rain falling steadily for two seconds, then a thunderclap at the'
                ' eight-second mark'
            ),
            (
                'rain falling steadily for six seconds, then a thunderclap at the'
                ' four-second mark'
            ),
            (
                'rain falling steadily for six seconds, then a thunderclap at the'
                ' seven-second mark'
            ),
        ],
    ),
    (
        'a low rumbling engine and a high-pitched bird chirping',
        [
            'a high-pitched engine and a low rumbling bird chirping',
            'a low rumbling engine and a high-pitched whistle',
            'a low rumbling engine, a high-pitched bird chirping, and a child laughing',
        ],
    ),
]


PERTURB_SPEECH_SHOTS = [
    (
        'The bell rang at fifteen past the hour and I forgot to lock the door.',
        [
            'The bell rang at fifty past the hour and I forgot to lock the door.',
            'The bill rang at fifteen past the hour and I forgot to lock the door.',
            'The bell rang at fifteen past the hour and I forgot to lock the drawer.',
        ],
    ),
    (
        'She said no, no, no to the offer on Tuesday.',
        [
            'She said no, no to the offer on Tuesday.',
            'She said no, no, no, no to the offer on Tuesday.',
            'She said no, no, no to the author on Tuesday.',
        ],
    ),
    (
        'Tell him again and again that the ferry leaves at four.',
        [
            'Tell him again that the ferry leaves at four.',
            'Tell him again and again that the fairy leaves at four.',
            'Tell them again and again that the ferry leaves at four.',
        ],
    ),
]


AQA_QUALITY_PROMPTS = {
    'audio': (
        'Listen to the audio. Is it a real and high-quality audio recording? '
        'Answer with only one word: yes or no.'
    ),
    'speech': (
        'Listen to the audio. Is it a real and high-quality human voice '
        'recording? Answer with only one word: yes or no.'
    ),
}


AQA_RELEVANCE_PROMPT = (
    'Does this audio contain the sound events described by the text: {text}? '
    'Please answer yes or no.'
)


AQA_RELEVANCE_SYSTEM = (
    'Your role is to listen attentively to the given audio and decide whether the'
    ' provided text accurately and completely describes what is heard.\n'
    'Make your judgment strictly based on the sounds in the audio --- do not '
    'guess, imagine, or add information that is not clearly audible.\n'
    'If something is missing, unclear, or uncertain, do not assume it exists.\n'
    'Your task: given an audio clip and a text description, carefully compare the'
    ' text with what you actually hear.\n'
    'Identify the main sound events (such as speech, background noise, music, or '
    'environmental sounds) and decide whether the text correctly reflects them.\n'
    'Always respond objectively and concisely, using "yes" or "no" to indicate '
    'whether the text matches the audio content.'
)


GEVAL_PROMPTS = {
    'audio_quality': (
        'Rate how much the audio sounds like a real and high-quality audio '
        'recording. Use an integer scale from 1 to 5, where 1 means very poor '
        'quality and 5 means excellent quality. Output only one integer: 1, 2, 3,'
        ' 4, or 5. Do not provide any explanation or additional text.\n'
    ),
    'speech_quality': (
        'Rate how much the audio sounds like a real and high-quality human voice '
        'recording. Use an integer scale from 1 to 5, where 1 means very poor '
        'quality and 5 means excellent quality. Output only one integer: 1, 2, 3,'
        ' 4, or 5. Do not provide any explanation or additional text.\n'
    ),
    'audio_relevance': (
        'Rate how closely the audio matches the sound events described by the '
        'following text. Use an integer scale from 1 to 5, where 1 means no match'
        ' and 5 means a perfect match. Output only one integer: 1, 2, 3, 4, or 5.'
        ' Do not provide any explanation or additional text.\n'
        '\n'
        'Text: {instruction}\n'
    ),
}


DIRECT_PROMPTS = {
    'audio_quality': (
        'Rate how much the audio sounds like a real and high-quality audio '
        'recording. Use a continuous scale from 1.00 to 5.00, where 1.00 means '
        'very poor quality and 5.00 means excellent quality. Output only one '
        'number with exactly two decimal places. Do not provide any explanation '
        'or additional text.\n'
    ),
    'speech_quality': (
        'Rate how much the audio sounds like a real and high-quality human voice '
        'recording. Use a continuous scale from 1.00 to 5.00, where 1.00 means '
        'very poor quality and 5.00 means excellent quality. Output only one '
        'number with exactly two decimal places. Do not provide any explanation '
        'or additional text.\n'
    ),
    'audio_relevance': (
        'Rate how closely the audio matches the sound events described by the '
        'following text. Use a continuous scale from 1.00 to 5.00, where 1.00 '
        'means no match and 5.00 means a perfect match. Output only one number '
        'with exactly two decimal places. Do not provide any explanation or '
        'additional text.\n'
        '\n'
        'Text: {instruction}\n'
    ),
}


COTTAGE_PROMPTS = {
    'REL': (
        'Rate the semantic alignment between this audio and the following text description on a scale from 1.00 to '
        '5.00, where 1.00 means no match and 5.00 means a perfect match. Think step by step, then give the final '
        'score. Output the score as a number rounded to two decimal places. Put your reasoning inside <think></think> '
        'and the final score inside <score></score>.\n\nText: {instruction}\n'
    ),
    'OVL': (
        'Rate the overall audio quality of this audio clip generated by a text-to-audio model on a scale from 1.00 to '
        '5.00, where 1.00 means very poor quality and 5.00 means excellent quality. Think step by step, then give the '
        'final score. Output the score as a number rounded to two decimal places. Put your reasoning inside '
        '<think></think> and the final score inside <score></score>.\n'
    ),
}
