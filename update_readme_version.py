"""

usage:
    python update_readme_version.py (bump2version --dry-run --list | grep new_version= | sed -r 's/^.*=//')
"""

import re

def update_version_in_readme(version):
    with open('README.md', 'r') as file:
        content = file.read()

    new_content = re.sub(r'version\s*=\s*"\d+\.\d+\.\d+"', f'version = "{version}"', content)

    with open('README.md', 'w') as file:
        file.write(new_content)

if __name__ == "__main__":
    import sys
    new_version = sys.argv[1]
    update_version_in_readme(new_version)


#