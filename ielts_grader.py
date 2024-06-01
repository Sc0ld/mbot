import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from textblob import TextBlob
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import math

class IELTSGrader:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()
        self.spell_checker = SpellChecker()
        self.transition_words = set([
            "additionally", "also", "besides", "furthermore", "moreover", "however", "instead", "nevertheless", "otherwise", 
            "consequently", "therefore", "thus", "accordingly", "hence", "similarly", "likewise", "conversely", "meanwhile", 
            "subsequently", "finally", "next", "then", "afterwards", "henceforth", "for example", "for instance", "such as", 
            "in conclusion", "to summarize", "in summary"
        ])
        self.common_words = self.stop_words | set([
            "be", "have", "do", "say", "get", "make", "go", "know", "take", "see", "come", "think", "look", "want", "give", 
            "use", "find", "tell", "ask", "work", "seem", "feel", "try", "leave", "call"
        ])

    def preprocess(self, text):
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word not in self.stop_words and word not in self.punctuation]
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in filtered_tokens]
        return lemmatized_tokens

    def suggest_corrections(self, tokens):
        suggestions = {}
        for word in tokens:
            if word not in self.spell_checker:
                suggestions[word] = self.spell_checker.correction(word)
        return suggestions

    def check_grammar(self, text):
        blob = TextBlob(text)
        corrected_text = str(blob.correct())
        grammar_errors = [
            {'original': original, 'corrected': corrected}
            for original, corrected in zip(text.split(), corrected_text.split())
            if original != corrected
        ]
        return grammar_errors

    def evaluate_task_response(self, essay, question):
        question_keywords = ' '.join(self.preprocess(question))
        essay_keywords = ' '.join(self.preprocess(essay))
        relevance_score = self.calculate_relevance_score(question_keywords, essay_keywords)

        # Map relevance score to IELTS band scale
        task_response_score = self.map_score_to_band(relevance_score, [(0.75, 7), (0.5, 6), (0.25, 5), (0, 4)])

        feedback = f"Task Response Score: {task_response_score}\n"
        if task_response_score < 5:
            feedback += "The essay does not sufficiently address the task prompt.\n"
        return task_response_score, feedback

    def calculate_relevance_score(self, question_keywords, essay_keywords):
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([question_keywords, essay_keywords])
        similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0]
        return similarity_score

    def evaluate_coherence_and_cohesion(self, essay):
        sentences = sent_tokenize(essay)
        num_sentences = len(sentences)
        
        # Count transition words and calculate logical flow
        transition_word_count = sum(1 for word in word_tokenize(essay.lower()) if word in self.transition_words)
        logical_flow = sum(1 for sentence in sentences if any(word in sentence.lower().split() for word in self.transition_words))

        # Coherence score
        coherence_score = min(7, 3 + transition_word_count + logical_flow // num_sentences)

        feedback = f"Coherence and Cohesion Score: {coherence_score}\n"
        feedback += f"Transition words used: {transition_word_count}\n"
        feedback += f"Sentences with logical flow: {logical_flow}/{num_sentences}\n"

        return coherence_score, feedback

    def evaluate_grammatical_range_and_accuracy(self, essay):
        grammar_errors = self.check_grammar(essay)
        
        # Grammar score
        grammar_score = self.map_score_to_band(len(grammar_errors), [(2, 7), (4, 6), (6, 5), (8, 4), (float('inf'), 3)])

        feedback = f"Grammatical Range and Accuracy Score: {grammar_score}\n"
        for error in grammar_errors:
            feedback += f"Error: '{error['original']}' corrected to '{error['corrected']}'\n"
        return grammar_score, feedback

    def evaluate_lexical_resource(self, essay):
        tokens = word_tokenize(essay.lower())
        unique_tokens = set(tokens)

        uncommon_words = [word for word in unique_tokens if word.isalpha() and word not in self.common_words and word not in self.stop_words]
        word_variety = len(set([word for word in tokens if word.isalpha()]))
        advanced_vocabulary = [word for word in tokens if word.isalpha() and word not in self.common_words and word not in self.stop_words]
        collocations = sum(1 for i in range(len(tokens)-1) if tokens[i] in self.common_words and tokens[i+1] not in self.common_words)
        lexical_diversity = len(uncommon_words) / len(tokens)

        lexical_score = self.map_score_to_band(lexical_diversity, [(0.25, 7), (0.20, 6), (0.15, 5), (0.10, 4), (0, 3)], additional_condition=(word_variety, [100, 80, 60, 40]))

        feedback = f"Lexical Resource Score: {lexical_score}\n"
        feedback += f"Uncommon words used: {len(uncommon_words)}\n"
        feedback += f"Total words: {len(tokens)}\n"
        feedback += f"Word variety: {word_variety}\n"
        feedback += f"Advanced vocabulary: {len(advanced_vocabulary)}\n"
        feedback += f"Collocations: {collocations}\n"
        feedback += f"Lexical diversity: {lexical_diversity:.2f}\n"

        return lexical_score, feedback

    def map_score_to_band(self, score, thresholds, additional_condition=None):
        for threshold, band in thresholds:
            if score >= threshold and (additional_condition is None or additional_condition[0] > additional_condition[1][thresholds.index((threshold, band))]):
                return band
        return 3

    def count_words(self, text):
        tokens = word_tokenize(text)
        words = [word for word in tokens if word.isalpha()]
        return len(words)

    def grade_essay(self, essay, question):
        essay_tokens = self.preprocess(essay)
        total_words = self.count_words(essay)

        misspelled_words = [word for word in essay_tokens if word not in self.spell_checker]
        num_spelling_errors = len(misspelled_words)
        spelling_corrections = self.suggest_corrections(misspelled_words)

        task_response_score, task_response_feedback = self.evaluate_task_response(essay, question)
        coherence_score, coherence_feedback = self.evaluate_coherence_and_cohesion(essay)
        lexical_score, lexical_feedback = self.evaluate_lexical_resource(essay)
        grammar_score, grammar_feedback = self.evaluate_grammatical_range_and_accuracy(essay)

        overall_score = (task_response_score + coherence_score + lexical_score + grammar_score) / 4

        feedback = ""
        if num_spelling_errors > 0:
            feedback += f"Spelling errors found. Suggestions: {spelling_corrections}\n"
        feedback += task_response_feedback
        feedback += coherence_feedback
        feedback += lexical_feedback
        feedback += grammar_feedback

        return {
            "word_count": total_words,
            "spelling_errors": num_spelling_errors,
            "spelling_corrections": spelling_corrections,
            "task_response_score": task_response_score,
            "coherence_score": coherence_score,
            "lexical_score": lexical_score,
            "grammar_score": grammar_score,
            "overall_score": overall_score,
            "feedback": feedback
        }

if __name__ == "__main__":
    grader = IELTSGrader()
    with open('new_essay.txt', 'r') as file:
        essay = file.read()
    question = "Some say that earning important quantities of cash as well as having fewer intervals of leisure is better than having less money with more free time. Discuss both views and state your opinion."

    result = grader.grade_essay(essay, question)
    for key, value in result.items():
        print(f"{key}: {value}")
