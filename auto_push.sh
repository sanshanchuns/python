#!/bin/sh

cd /Users/leo/Documents/workspace-test/python
git add .
git ci -m"$(uname -n) $(date) $line"
git pull --rebase
git push