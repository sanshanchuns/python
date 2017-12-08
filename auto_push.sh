#!/bin/sh

cd /Users/leo/Documents/workspace-test/python
git add .
git ci -m"auto_push_commit"
git pull --rebase
git push