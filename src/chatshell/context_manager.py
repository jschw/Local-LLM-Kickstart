import os
from datetime import datetime

class ContextManager:
    """
    Context manager for handling RAG context and RAG provider.
    """
    def __init__(self):
        from .vectorstore import ChatshellVectorsearch
        self.doc_base_dir = None

        self.context_list = []
        self.context_update_time = ""

        self.rag_provider = ChatshellVectorsearch()
        self.rag_content_list = []
        self.rag_mode = 0  # 0 = file, 1 = web, 2 = clipboard
        self.rag_update_time = ""

    def set_doc_base_dir(self, doc_base_dir):
        self.doc_base_dir = doc_base_dir

    def rag_update_file(self, document_path) -> list:
        # Split paths if more than one
        document_paths_arg = document_path.split(";")

        # Check document exist
        document_paths_exist = []
        output_list = []

        for doc in document_paths_arg:
            doc_current = doc
            if not os.path.isfile(doc_current):
                # Document is not available at absolute path, checking rel. path
                doc_current = os.path.join(self.doc_base_dir, doc_current)
                if not os.path.isfile(doc_current):
                    # Document is not available -> return error
                    print(f"--> Document {doc_current} not found.")
                    output_list.append(f"Document {doc_current} not found.")
                    continue

            output_list.append(f"Document {doc_current} existing and added to RAG document list.")
            document_paths_exist.append(doc_current)

        if len(document_paths_exist) == 0:
            print("--> No existing document found at given path.")
            output_list.insert(0, "No existing document found at given path.")
            return [False, "\n".join(output_list)]

        # Update RAG
        self.rag_content_list = document_paths_exist
        self.rag_update_time = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        self.rag_mode = 0
        rag_update_ok = self.rag_provider.init_vectorstore_pdf(document_paths_exist)

        if rag_update_ok:
            output_list.insert(0, "Ready, you can now chat with your document(s)!")
        else:
            output_list.insert(0, "There was a problem updating the RAG system. Please try again.")

        return [rag_update_ok, "\n".join(output_list)]

    def rag_update_web(self, url, deep):
        # Split paths if more than one
        urls = url.split(";")
        self.rag_update_time = datetime.now().strftime('%d-%m-%Y %H:%M:%S')

        # Update RAG
        self.rag_mode = 1
        self.rag_content_list = urls
        rag_update_ok = self.rag_provider.init_vectorstore_web(urls, deep)

        return rag_update_ok
    
    def rag_update_clipbrd(self, input):
        rag_update_ok = self.rag_provider.init_vectorstore_str(input)
        self.rag_mode = 2

        return rag_update_ok

    def __enter__(self):
        # Optionally, could accept initial context here
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset_context()

    def add_context(self, input):
        self.context_update_time = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        self.context_list.append(input)

    def get(self):
        context = ""
        for i in self.context_list:
            context += f"{{\n{i}\n}},\n"
        return context

    def reset_context(self):
        self.context_list = []

    def reset_rag_context(self):
        self.rag_content_list = []