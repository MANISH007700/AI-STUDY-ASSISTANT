import json
import os
import time
import uuid
from typing import Any, Dict, List

from openai import OpenAI, AuthenticationError
from pymilvus.exceptions import ConnectionNotExistException

from vector_store import StudyMaterialsStore

class StudyAssistant:
    def __init__(self):
        try:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # Test the client to ensure the API key is valid
            self.client.models.list()
        except AuthenticationError:
            raise ValueError("Invalid OpenAI API key provided")
        except Exception as e:
            raise Exception(f"Error initializing OpenAI client: {str(e)}")
        
        self.model = "gpt-4o"
        self.conversation_history = []
        self.vector_store = None
        self.setup_vector_store()

    def setup_vector_store(self):
        """Initialize or reinitialize the vector store."""
        try:
            unique_id = str(uuid.uuid4())[:8]
            timestamp = str(int(time.time()))[-6:]
            db_uri = f"tmp/sa_{unique_id}_{timestamp}.db"
            self.vector_store = StudyMaterialsStore(
                collection="study_materials", dimension=1536, uri=db_uri
            )
        except Exception as e:
            print(f"Error setting up vector store: {e}")
            unique_id = str(uuid.uuid4())[:8]
            timestamp = str(int(time.time()))[-6:]
            new_uri = f"tmp/sa_{unique_id}_{timestamp}.db"
            self.vector_store = StudyMaterialsStore(
                collection="study_materials", dimension=1536, uri=new_uri
            )

    def get_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small", input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise

    def store_material(self, text: str, metadata: Dict = None) -> int:
        """Store study material in the vector store."""
        try:
            embedding = self.get_embedding(text)
            meta = metadata or {}
            return self.vector_store.store_vectors([text], [embedding], [meta])[0]
        except ConnectionNotExistException:
            print("Reconnecting to vector store...")
            self.setup_vector_store()
            embedding = self.get_embedding(text)
            meta = metadata or {}
            return self.vector_store.store_vectors([text], [embedding], [meta])[0]
        except Exception as e:
            print(f"Error storing material: {e}")
            raise

    def search_materials(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant study materials."""
        try:
            query_embedding = self.get_embedding(query)
            return self.vector_store.search_vectors(query_embedding, top_k)
        except ConnectionNotExistException:
            print("Reconnecting to vector store...")
            self.setup_vector_store()
            query_embedding = self.get_embedding(query)
            return self.vector_store.search_vectors(query_embedding, top_k)
        except Exception as e:
            print(f"Error searching materials: {e}")
            raise

    def generate_response(self, query: str) -> str:
        """Generate a response to a student's question."""
        try:
            relevant_materials = self.search_materials(query)
            context = "\n".join([item["text"] for item in relevant_materials])
            history = "\n".join(
                [f"Student: {msg['user']}\nAssistant: {msg['assistant']}"
                 for msg in self.conversation_history[-5:]]
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful study assistant. Use the provided context to answer questions accurately and clearly.",
                    },
                    {"role": "system", "content": f"Study materials:\n{context}"},
                    {"role": "system", "content": f"Previous conversation:\n{history}"},
                    {"role": "user", "content": query},
                ],
                temperature=0.7,
            )
            self.conversation_history.append(
                {"user": query, "assistant": response.choices[0].message.content}
            )
            return response.choices[0].message.content
        except ConnectionNotExistException:
            print("Reconnecting to vector store...")
            self.setup_vector_store()
            return self.generate_response(query)
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error: {str(e)}"

    def generate_flashcards(self, topic: str, count: int = 5) -> List[Dict[str, str]]:
        """Generate flashcards for a specific topic."""
        try:
            relevant_materials = self.search_materials(topic)
            context = "\n".join([item["text"] for item in relevant_materials])
            prompt = f"""
            Create {count} flashcards about {topic} based on the study materials.
            Each flashcard should have a question and answer.
            
            Study materials:
            {context}
            
            Return a JSON array of objects with 'question' and 'answer' fields.
            Questions should be challenging but fair, answers concise but complete.
            """
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a study assistant creating effective flashcards."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
            content = response.choices[0].message.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            return json.loads(content)
        except json.JSONDecodeError:
            return [{"question": "Error", "answer": "Could not generate flashcards."}]
        except Exception as e:
            print(f"Error generating flashcards: {e}")
            return [{"question": "Error", "answer": f"Error: {str(e)}"}]

    def generate_quiz(self, topic: str, num_questions: int = 5) -> List[Dict[str, Any]]:
        """Generate a quiz with multiple-choice questions."""
        try:
            relevant_materials = self.search_materials(topic)
            context = "\n".join([item["text"] for item in relevant_materials])
            prompt = f"""
            Create a quiz with {num_questions} multiple-choice questions about {topic}.
            Each question should have 4 options with one correct answer.
            
            Study materials:
            {context}
            
            Return a JSON array of objects with:
            - 'question': question text
            - 'options': array of 4 answers
            - 'correct_index': index (0-3) of correct answer
            - 'explanation': brief explanation
            
            Questions should be challenging but fair.
            """
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a study assistant creating effective quizzes."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
            content = response.choices[0].message.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            return json.loads(content)
        except json.JSONDecodeError:
            return [{"question": "Error", "options": ["Error"], "correct_index": 0, "explanation": "Could not generate quiz."}]
        except Exception as e:
            print(f"Error generating quiz: {e}")
            return [{"question": "Error", "options": ["Error"], "correct_index": 0, "explanation": f"Error: {str(e)}"}]

    def summarize_material(self, text: str, max_length: int = 500) -> str:
        """Generate a concise summary of study material."""
        try:
            prompt = f"""
            Summarize the following study material clearly, highlighting key concepts.
            Keep the summary under {max_length} characters.
            
            Study material:
            {text}
            """
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a study assistant creating concise summaries."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Error: {str(e)}"

    def explain_concept(self, concept: str) -> str:
        """Provide a detailed explanation of a concept."""
        try:
            relevant_materials = self.search_materials(concept)
            context = "\n".join([item["text"] for item in relevant_materials])
            prompt = f"""
            Explain "{concept}" in detail using the study materials.
            If materials are insufficient, use general knowledge.
            
            Study materials:
            {context}
            
            Explanation should be:
            1. Clear and understandable
            2. Include examples
            3. Break down complex ideas
            4. Highlight key points
            """
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a study assistant explaining concepts clearly."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error explaining concept: {e}")
            return f"Error: {str(e)}"

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.conversation_history