from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("/articles/Short_Novels_Introduction.pdf")
documents = loader.load()

# Search for the text in all pages
for i, doc in enumerate(documents):
    if "February 19" in doc.page_content or "9066" in doc.page_content:
        print(f"Found on page {i+1}")
        print(doc.page_content)
        break
else:
    print("Text not found in any page - page is likely a scanned image")