module.exports = {
    root: true,
    env: {browser: true, es2020: true},
    extends: [
        'eslint:recommended',
        'plugin:@typescript-eslint/recommended',
        'plugin:react-hooks/recommended',
    ],
    settings: {
        'import/parsers': {
            '@typescript-eslint/parser': ['.ts', '.tsx'],
        },
    },
    ignorePatterns: ['dist', '.eslintrc.cjs'],
    parser: '@typescript-eslint/parser',
    plugins: ['react-refresh'],
    rules: {
        'react-refresh/only-export-components': [
            'warn',
            {allowConstantExport: true},

        ],
        'import/no-extraneous-dependencies': 'off',
        // '@typescript-eslint/await-thenable': 'error',
        // '@typescript-eslint/no-floating-promises': ['error', {ignoreVoid: true, ignoreIIFE: true}],
        // '@typescript-eslint/no-misused-promises': ['error', {
        //     checksConditionals: true,
        //     checksSpreads: true,
        //     checksVoidReturn: false
        // }],
        '@typescript-eslint/no-shadow': 'error',
        // 'require-await': 'off',
        // '@typescript-eslint/require-await': ['error'],
        '@typescript-eslint/no-unused-vars': ['warn', {argsIgnorePattern: '^_', varsIgnorePattern: '^_'}],
        // 'comma-dangle': ['error', 'always-multiline'],  // May conflict with prettier
        'curly': ['error', 'all'],
        'import/extensions': ['warn', 'never', {css: 'always', scss: 'always', json: 'always'}],
        'import/no-import-module-exports': 'off',
        'import/no-unresolved': 'error',
        'max-classes-per-file': 'off',
        'no-shadow': 'off',
        'no-unused-vars': 'off',
        'no-use-before-define': 'off',
        'prefer-const': 'warn',
        'prettier/prettier': 'off',
        'react/jsx-filename-extension': ['warn', {extensions: ['.tsx']}],
        'react/react-in-jsx-scope': 'off',
        "no-plusplus": 'off',
        "no-underscore-dangle": 'off',
        "react/function-component-definition": 0,
        "sort-imports": ["error", {
            "ignoreDeclarationSort": true
        }],
        "import/first": "error",
        "import/newline-after-import": "error",
        "import/no-duplicates": "error",
        "unused-imports/no-unused-imports": "error",
        "unused-imports/no-unused-vars": [
            "warn",
            {vars: "all", varsIgnorePattern: "^_", args: "after-used", argsIgnorePattern: "^_"}
        ]
    },
}
