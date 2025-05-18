import logging
import datasets
from langchain.docstore.document import Document

def load_and_prepare_docs():
    logging.info("Loading dataset from Hugging Face Hub...")
    guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")
    logging.info(f"Dataset loaded with {len(guest_dataset)} entries.")
    docs = [
        Document(
            page_content="\n".join([
                f"Name: {guest['name']}",
                f"Relation: {guest['relation']}",
                f"Description: {guest['description']}",
                f"Email: {guest['email']}"
            ]),
            metadata={"name": guest["name"]}
        )
        for guest in guest_dataset
    ]
    logging.info(f"Created {len(docs)} Document objects.")
    return docs

if __name__ == "__main__":
    prepared_docs = load_and_prepare_docs()
    logging.info(f"Successfully loaded and prepared {len(prepared_docs)} documents.")
    if prepared_docs:
        logging.info("\nSample Document:")
        logging.info(prepared_docs[0])
