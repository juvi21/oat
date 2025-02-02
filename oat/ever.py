import os

def create_markdown_from_files(root_dir):
    # Files/directories to ignore
    ignore_list = ['.git', '.pytest_cache', '__pycache__', 'LICENSE', 'README.md']
    
    # Dictionary to store file extensions and their corresponding language identifiers and comments
    lang_map = {
        '.py': ('python', '#'),
        '.cpp': ('cpp', '//'),
        '.h': ('cpp', '//'),
        '.js': ('javascript', '//'),
        '.html': ('html', '<!--'),
        '.css': ('css', '/*'),
        '.java': ('java', '//'),
        # Add more mappings as needed
    }
    
    markdown_content = []
    
    for root, dirs, files in os.walk(root_dir):
        # Remove ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_list]
        
        for file in files:
            if file not in ignore_list:
                file_path = os.path.join(root, file)
                # Get file extension
                _, ext = os.path.splitext(file)
                # Get language identifier and comment symbol from extension, default to 'text' and '#'
                lang_info = lang_map.get(ext, ('text', '#'))
                lang = lang_info[0]
                comment_symbol = lang_info[1]
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Create markdown block with file path as comment
                    markdown_content.append(
                        f"```{lang}\n{comment_symbol} {file_path}\n{content}\n```\n"
                    )
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    # Write to markdown file
    with open('files_content.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(markdown_content))

if __name__ == "__main__":
    # Use current directory as root
    create_markdown_from_files('.')