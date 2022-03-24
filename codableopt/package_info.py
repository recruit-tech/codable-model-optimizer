# (major, minor, patch, prerelease)
VERSION = (0, 1, 1, '')
__shortversion__ = '.'.join(map(str, VERSION[:3]))
__version__ = '.'.join(map(str, VERSION[:3])) + ''.join(VERSION[3:])

__package_name__ = 'codableopt'
__author_names__ = 'Tomomitsu Motohashi, Kotaro Tanahashi'
__author_emails__ = 'tomomoto1983@gmail.com, tanahashi@r.recruit.co.jp'
__maintainer_names__ = 'Kotaro Tanahashi'
__maintainer_emails__ = 'tanahashi@r.recruit.co.jp'
__homepage__ = ''
__repository_url__ = 'https://github.com/recruit-tech/codable-model-optimizer'
__download_url__ = 'https://github.com/recruit-tech/codable-model-optimizer'
__description__ = 'Optimization problem meta-heuristics solver for easy modeling.'
__license__ = 'Apache 2.0'
__keywords__ = 'optimization, solver, modeling, meta-heuristics, mathematical optimization'
