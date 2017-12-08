#!/bin/sh

git add .
git ci -m"auto_push_commit"
git pull
git push