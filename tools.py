#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small, generic helper functions.

- pause_commandline(): prompt user to continue (when outside Spyder).
"""
import os

def pause_commandline(msg='Press Enter to continue.'):
    """Prompt user to continue (when outside Spyder).

    Only do this script appears to be run from command line.
    This is to prevent plot windows from closing at program termination.
    """

    if 'SPYDER_ARGS' not in os.environ:
        input(msg)


