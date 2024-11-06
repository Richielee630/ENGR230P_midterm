import re

def make_chapters_foldable(input_file, output_file):
    with open(input_file, 'r') as file:
        content = file.readlines()

    # Initialize variables to store transformed content and chapter counter
    modified_content = []
    chapter_counter = 1
    inside_chapter = False

    for line in content:
        # Detect start of a new chapter based on your chapter header pattern
        if re.match(r"# Quiz for Document \d+: .*", line):
            # Close the previous chapter's details if one is open
            if inside_chapter:
                modified_content.append("</details>\n")
            
            # Start a new foldable section
            chapter_title = f"Chapter {chapter_counter}: {line.strip()[21:]}"
            modified_content.append(f"<details>\n  <summary>{chapter_title}</summary>\n\n")
            modified_content.append(line)
            
            inside_chapter = True
            chapter_counter += 1
        else:
            modified_content.append(line)
    
    # Close the last chapter's details tag
    if inside_chapter:
        modified_content.append("</details>\n")

    # Write the modified content to the output file
    with open(output_file, 'w') as file:
        file.writelines(modified_content)

    print(f"Foldable chapters added and saved to {output_file}.")

# Usage example
make_chapters_foldable('230P_Midterm_practice_exam.md', 'foldable_quiz_content.md')
