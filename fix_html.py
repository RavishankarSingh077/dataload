
import os

path = r"s:\HOME\stock prediction\nasdaq_model\templates\index.html"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
skip = 0
for line in lines:
    if skip > 0:
        skip -= 1
        continue
    if '<div class="chat-hint">Ask me anything</div>' in line:
        # Found the start of the block to replace
        # Use simple reliable replacement
        new_lines.append('        <div style="display: flex; align-items: center; gap: 15px;">\n')
        new_lines.append('            <div class="chat-hint">Ask me anything</div>\n')
        new_lines.append('            <div class="chat-bubble" onclick="toggleChat(true)">Chatbot</div>\n')
        new_lines.append('        </div>\n')
        skip = 3 # Skip original bubble lines
    else:
        new_lines.append(line)

with open(path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
print("File updated successfully.")
