public static class StringExtensions
{
    public static uint ToFNV32(this string s)
    {
        uint h = 2166136261U;
        for (int i = 0; i != s.Length; i++)
            h = (h ^ s[i]) * 16777619U;
        return h;
    }
    private static char[] str;
    private static int k, k0, j;
    static private void SetTo(string s)
    {
        for (int i = 0; i != s.Length; ++i)
            str[j + 1 + i] = s[i];
        k = j + s.Length;
    }
    static private bool IsCons(int i)
    {
        switch (str[i])
        {
            case 'a':
            case 'e':
            case 'i':
            case 'o':
            case 'u':
                return false;
            case 'y':
                return (i == k0) ? true : IsCons(i - 1) == false;
            default:
                return true;
        }
    }
    static private bool HasVowel()
    {
        for (int i = k0; i <= j; i++)
            if (IsCons(i) == false) return true;
        return false;
    }
    static private bool DoubleC(int j)
    {
        if (j < k0 + 1) return false;
        if (str[j] != str[j - 1]) return false;
        return IsCons(j);
    }
    static private int CountConsSeq()
    {
        int n = 0;
        int i = k0;
        while (true)
        {
            if (i > j) return n;
            if (IsCons(i) == false) break;
            i++;
        }
        i++;
        while (true)
        {
            while (true)
            {
                if (i > j) return n;
                if (IsCons(i)) break;
                i++;
            }
            i++;
            n++;
            while (true)
            {
                if (i > j) return n;
                if (!IsCons(i)) break;
                i++;
            }
            i++;
        }
    }
    static private bool CVC(int i)
    {
        if (i < k0 + 2 || IsCons(i) == false || IsCons(i - 1) || IsCons(i - 2) == false) return false;
        if (str[i] == 'w' || str[i] == 'x' || str[i] == 'y') return false;
        return true;
    }
    static private bool Ends(string s)
    {
        if (s.Length > k - k0 + 1) return false;
        for (int i = 0; i != s.Length; ++i)
            if (str[k - s.Length + 1 + i] != s[i]) return false;
        j = k - s.Length;
        return true;
    }
    static private void Step1AB()
    {
        if (str[k] == 's')
        {
            if (Ends("sses")) k = k - 2;
            else
                if (Ends("ies")) SetTo("i");
                else
                    if (str[k - 1] != 's') k = k - 1;
        }
        if (Ends("eed")) { if (CountConsSeq() > 0) k = k - 1; }
        else
            if ((Ends("ed") || Ends("ing")) && HasVowel())
            {
                k = j;
                if (Ends("at")) SetTo("ate");
                else
                    if (Ends("bl")) SetTo("ble");
                    else
                        if (Ends("iz")) SetTo("ize");
                        else
                            if (DoubleC(k))
                            {
                                k = k - 1;
                                if (str[k] == 'l' || str[k] == 's' || str[k] == 'z') k = k + 1;
                            }
                            else if (CountConsSeq() == 1 && CVC(k)) SetTo("e");
            }
    }
    static private void Step1C()
    {
        if (Ends("y") && HasVowel()) str[k] = 'i';
    }
    static private void R(string s)
    {
        if (CountConsSeq() > 0) SetTo(s);
    }
    static private void Step2()
    {
        switch (str[k - 1])
        {
            case 'a':
                if (Ends("ational")) { R("ate"); break; }
                if (Ends("tional")) { R("tion"); break; }
                break;
            case 'c':
                if (Ends("enci")) { R("ence"); break; }
                if (Ends("anci")) { R("ance"); break; }
                break;
            case 'e':
                if (Ends("izer")) { R("ize"); break; }
                break;
            case 'g':
                if (Ends("logi")) { R("log"); break; }
                break;
            case 'l':
                if (Ends("bli")) { R("ble"); break; }
                if (Ends("alli")) { R("al"); break; }
                if (Ends("entli")) { R("ent"); break; }
                if (Ends("eli")) { R("e"); break; }
                if (Ends("ousli")) { R("ous"); break; }
                break;
            case 'o':
                if (Ends("ization")) { R("ize"); break; }
                if (Ends("ation")) { R("ate"); break; }
                if (Ends("ator")) { R("ate"); break; }
                break;
            case 's':
                if (Ends("alism")) { R("al"); break; }
                if (Ends("iveness")) { R("ive"); break; }
                if (Ends("fulness")) { R("ful"); break; }
                if (Ends("ousness")) { R("ous"); break; }
                break;
            case 't':
                if (Ends("aliti")) { R("al"); break; }
                if (Ends("iviti")) { R("ive"); break; }
                if (Ends("biliti")) { R("ble"); break; }
                break;
        }
    }
    static private void Step3()
    {
        switch (str[k])
        {
            case 'e':
                if (Ends("icate")) { R("ic"); break; }
                if (Ends("ative")) { R(""); break; }
                if (Ends("alize")) { R("al"); break; }
                break;
            case 'i':
                if (Ends("iciti")) { R("ic"); break; }
                break;
            case 'l':
                if (Ends("ical")) { R("ic"); break; }
                if (Ends("ful")) { R(""); break; }
                break;
            case 's':
                if (Ends("ness")) { R(""); break; }
                break;
        }
    }
    static private void Step4()
    {
        switch (str[k - 1])
        {
            case 'a':
                if (Ends("al")) break;
                return;
            case 'c':
                if (Ends("ance")) break;
                if (Ends("ence")) break;
                return;
            case 'e':
                if (Ends("er")) break;
                return;
            case 'i':
                if (Ends("ic")) break;
                return;
            case 'l':
                if (Ends("able")) break;
                if (Ends("ible")) break;
                return;
            case 'n':
                if (Ends("ant")) break;
                if (Ends("ement")) break;
                if (Ends("ment")) break;
                if (Ends("ent")) break;
                return;
            case 'o':
                if (Ends("ion") && j >= 0 && (str[j] == 's' || str[j] == 't')) break;
                if (Ends("ou")) break;
                return;
            case 's':
                if (Ends("ism")) break;
                return;
            case 't':
                if (Ends("ate")) break;
                if (Ends("iti")) break;
                return;
            case 'u':
                if (Ends("ous")) break;
                return;
            case 'v':
                if (Ends("ive")) break;
                return;
            case 'z':
                if (Ends("ize")) break;
                return;
            default:
                return;
        }
        if (CountConsSeq() > 1) k = j;
    }
    static private void Step5()
    {
        j = k;
        if (str[k] == 'e')
        {
            int a = CountConsSeq();
            if (a > 1 || (a == 1 && CVC(k - 1) == false)) k--;
        }
        if (str[k] == 'l' && DoubleC(k) && CountConsSeq() > 1) k--;
    }
    public static string ToPorterStem(this string s)
    {
        if (s.Length < 4) return s;
        str = s.ToCharArray(0, s.Length);
        k0 = 0;
        k = s.Length - 1;
        Step1AB();
        Step1C();
        Step2();
        Step3();
        Step4();
        Step5();
        return s.Substring(0, k + 1);
    }
}
