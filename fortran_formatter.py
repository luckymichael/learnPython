# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 03:14:20 2016

@author: Michael
"""
from string import *

n_indent = 2

f90old = r'D:\Cloud\Dropbox\postdoc\unswat\scr\unswat\sol_scr\solflw-ross-mod.F90'
f90new = r'D:\Cloud\Dropbox\postdoc\unswat\scr\unswat\sol_scr\solflw-ross-mod.F90.new'

#f90old = r'/home/mgou/Dropbox/postdoc/unswat/scr/unswat/sol_scr/solflw-ross-mod.f90'
#f90new = r'/home/mgou/Dropbox/postdoc/unswat/scr/unswat/sol_scr/solflw-ross-mod.f90.new'


# fortran keywords
f77keys = 'assign, backspace, block data, call, close, common, continue, data, dimension, do, else, else if, end, endfile, endif, entry, equivalence, external, format, function, goto, if, implicit, none, inquire, intrinsic, open, parameter, pause, print, program, read, return, rewind, rewrite, save, stop, subroutine, then, write'
f90keys = 'allocatable, allocate, case, contains, cycle, deallocate, elsewhere, exit, include, interface, intent, module, namelist, nullify, only, operator, optional, pointer, private, procedure, public, recursive, result, select, sequence, target, type, use, while, where'
f95keys = 'elemental, forall, pure'
f03keys = 'abstract, associate, asynchronous, bind, class, deferred, enum, enumerator, extends, final, flush, generic, import, non_overridable, nopass, pass, protected, value, volatile, wait'
f08keys = 'block, codimension, do concurrent, contiguous, critical, error stop, submodule, sync all, sync images, sync memory, lock, unlock'
fkeys = f77keys + ', ' + f90keys + ', ' + f95keys + ', ' + f03keys + ', ' + f08keys
fkey  = []
for k in  fkeys.split(','): fkey.append(k.strip())
block_keys_begin = ['do','if','where','select','while','associate','abstract','subroutine','function','module','forall','type ']
block_keys_middle = 'else,case,type is,typeis,contains'   
block_keys_end = 'end'
declaration_keys = ['integer','character','real','doubleprecision','double precision','logical','complex','type(','use']

fkey.extend(declaration_keys)

#%% 
def indent_space(indent):
    return ' '*n_indent*indent

def word_is_quoted(line,word):
    iloc = line.find(word)
    if iloc < 0:
        # word not in line
        return False
    else:
        if line[:iloc].count("'") % 2 == 1 or line[:iloc].count('"') % 2 == 1:
            return True
        else:
            return False
            
def endswithfirstbracketpair(line):
    s = line.strip()
    l = s.rfind('(')
    r = s[l:].find(')')+l+1
    while r < len(s) - 1 and r > -1:
        s = s[:l]+'_'+s[l+1:r-1]+'_'+ s[r:]
        l = s.rfind('(')
        r = s[l:].find(')')+l+1
    return r == len(s)
    
def lowerkeywords(line):
    for key in fkey:
        loc = line.lower().find(key)
        if loc >= 0:
            if loc > 0:
                if line[loc-1] in ascii_letters or line[loc-1] == '_': 
                    continue                    
            if loc + len(key) < len(line):
                if line[loc+len(key)] in digits or line[loc+len(key)] in ascii_letters or line[loc+len(key)] == '_': 
                    continue
                    
            line = line[:loc] + key + line[loc+len(key):]
    return line
#%% 

linenum = 0
indent = 0
linemax = 0
continueline = 0
labelnum = ''
fold = open(f90old, 'r', encoding='UTF-8')
fnew = open(f90new, 'w', encoding='UTF-8')    
for line in fold:
    linenum += 1
    if line.startswith('#'): 
        fnew.write(line)
        continue
    ## get the indent space    
    head = indent_space(indent+continueline)
    
    ## split code and comment
    printed=False
    line = line.strip() ## remove leading space and line break at the end
    if (len(line) == 0):
        fnew.write('\n')
        continue
    
    commentLoc = line.find('!')
    if commentLoc >= 0:
        if commentLoc == 0:
            comments = line[commentLoc:]
        else:
            comments = ' ! ' + line[commentLoc+1:].lstrip()
        line = line[:(commentLoc)].rstrip()
    else:
        comments = ''
    
    line = lowerkeywords(line)
    ## check block name
    words = line.split()
    blockName = ''
    if len(words) > 2:
        if words[0].count(':') + words[1].count(':') == 1 and words[0].count('(') == 0 and words[1].count('(') == 0:
            blockLoc = line.find(':')        
            blockName = line[:(blockLoc+1)] + ' '
            line = line[(blockLoc+1):].lstrip()
        
    ## check if it's labeled
    if line != '':
        if (line[0].isdigit()):
            blockName = words[0]
            line = line.lstrip(blockName)
        
    ## check continuation
    if (line.endswith('&')): 
        continueline = 1
    else: 
        continueline = 0 
        
    ## check indention
    
    # if indent block, check if it's an one-liner 
    lowerline = line.lower()
    ## check if it's variable declaration        
    #for key in block_keys_begin:
    if linenum == 2472 and linenum < 2500:
        print(lowerline, indent)
    # blocks that must end with 'end'    
    if lowerline.count('subroutine') > 0 and (not lowerline.startswith('end')) and (not word_is_quoted(lowerline,'subroutine')): indent += 1
    if lowerline.count('function') > 0 and (not lowerline.startswith('end')) and (not word_is_quoted(lowerline,'function')): indent += 1   # elemental real function 
    if lowerline.count('interface') > 0 and (not lowerline.startswith('end')) and (not word_is_quoted(lowerline,'interface')): indent += 1
    if lowerline.startswith('select')  and lowerline.count('end') == 0: indent += 1    
    if lowerline.startswith('associate')  and lowerline.count('end') == 0: indent += 1   
    if lowerline.startswith('abstract')  and lowerline.count('end') == 0: indent += 1  
    if (lowerline.startswith('do ') or lowerline == 'do')  and lowerline.count('end') == 0: indent += 1    
    if lowerline.startswith('module') and lowerline.count('end') == 0: indent += 1
        
    if lowerline.startswith('if')  and lowerline.endswith('then'): indent += 1
        
    if lowerline.startswith('where')  and endswithfirstbracketpair(lowerline): indent += 1  
    if lowerline.startswith('forall')  and endswithfirstbracketpair(lowerline): indent += 1  
    if lowerline.startswith('type '): 
        if lowerline.startswith('type is'):
            head = indent_space(indent-1) 
        else:
            indent += 1                
    
    if lowerline.startswith('else'): head = indent_space(indent-1)
    if lowerline.startswith('case'): head = indent_space(indent-1)
    if lowerline.startswith('typeis'): head = indent_space(indent-1)  
    if lowerline.startswith('contains'): head = indent_space(indent-1)     
        
    if lowerline.startswith('end'): indent -= 1; head = indent_space(indent)  
        
    ## check if it's variable declaration        
    for key in declaration_keys:
        ## variable-type,intent(xxxxx),parameter,dimension(xxx),allocatable,pointer
        if lowerline.startswith(key):
            # split definition and name
            if line.startswith('use'): 
                if line.count('only')>0:
                    sep = ':'
                else:
                    break
            else:
                sep = '::'
            
            nameLoc = line.find(sep)    
            if nameLoc < 1: 
                print('No :: at line: ' + line)
                break
            else:                
                names = line[nameLoc:]
                line = line[:(nameLoc)]
                        
            
            
            # divide declaration
            words = line.split(',')
            line = words[0].strip().ljust(19)
            for word in words[1:]:
                line = line + ',' + word.strip().ljust(19)
            
            linemax=max(linemax,len(line.strip()))
            
            
            # check the variiable name
            if names.startswith(sep): names = names[len(sep):].strip()
            
            # check if it has multiple variables
            newline = ''                
            for nam in names.split(','):
                varname, eq, varval = nam.partition('=')
                newline = (head + line).rstrip().ljust(60) + sep.ljust(3) + (varname.strip().ljust(10) + eq + ' ' + varval.strip()).ljust(20) + comments
                if not newline.endswith('\n'): newline = newline + '\n'
                fnew.write(newline)
            printed = True
    
    if not printed:
        line = line + comments
        if not line.endswith('\n'): line = line + '\n'
        head = blockName + head[min(len(blockName),len(head)):]
        fnew.write(head + line)
            
            
fold.close()
fnew.close()