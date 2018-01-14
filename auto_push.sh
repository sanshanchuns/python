#!/bin/sh

cd /Users/leo/Documents/workspace-test/python
git add .
git ci -m"$(date) $line"
git pull --rebase
git push