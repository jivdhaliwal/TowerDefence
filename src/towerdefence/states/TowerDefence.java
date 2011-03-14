/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package towerdefence.states;

import org.newdawn.slick.*;
import org.newdawn.slick.state.StateBasedGame;

public class TowerDefence extends StateBasedGame
{
     static int height = 18*32;
     static int width = 18*32;

     static boolean fullscreen = false;

     static boolean showFPS = true;

     static String title = "Basic Game Template";

     static int fpslimit = 60;

     public static final int MAINMENUSTATE = 0;
     public static final int GAMEPLAYSTATE = 1;

     

     public TowerDefence(String title)
     {
          super(title);

          //this.addState(new MainMenuState(MAINMENUSTATE));
          this.addState(new GameplayState(GAMEPLAYSTATE));
          this.enterState(GAMEPLAYSTATE);
     }

     public static void main(String[] args) throws SlickException
     {
          AppGameContainer app = new AppGameContainer(new ScalableGame(new TowerDefence(title),width,height));
           //=new AppGameContainer(new TowerDefence(title));

          app.setDisplayMode((int)(width), (int)(height), fullscreen);
          app.setSmoothDeltas(true);
          app.setTargetFrameRate(fpslimit);
          app.setShowFPS(showFPS);
          app.start();
     }

    @Override
    public void initStatesList(GameContainer container) throws SlickException {
        //this.getState(MAINMENUSTATE).init(container, this);
        this.getState(GAMEPLAYSTATE).init(container, this);
    }
}